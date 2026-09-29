# =============================================================================
# scrambling_map_qp.jl — QuantumPropagators.jl backend + automatic selection.
#
# Include AFTER scrambling_map.jl. Enclosing module needs:
#     using QuantumPropagators
#     using ExponentialUtilities   # only for method = ExponentialUtilities
# Uses from scrambling_map.jl: AbstractStatePropagatorAlg, DiagonalizationPropagatorAlg,
# _operator, _tvec.
# All numerics (Chebychev coefficients, spectral range, error control) are done by
# QuantumPropagators; this file maps output times onto QP grids and picks the method.
# =============================================================================

import QuantumPropagators
const QP = QuantumPropagators

# -----------------------------------------------------------------------------
# 1. QPAlg: thin wrapper around QP.init_prop / prop_step!
# -----------------------------------------------------------------------------
"""
    QPAlg(method = QP.Cheby; max_dt = Inf, block = false, init_prop_kwargs...)

Propagation with QuantumPropagators.jl; `method` and remaining kwargs go to `QP.init_prop`:

    QPAlg()                                                       # Chebychev, QP spectral range
    QPAlg(QP.Cheby; cheby_coeffs_limit = 1e-14)
    QPAlg(QP.Cheby; specrange_method = :manual, E_min = -5.0, E_max = 5.0)
    QPAlg(QP.Newton; max_dt = 0.5, m_max = 10, relerr = 1e-12)
    QPAlg(QP.ExpProp; convert_operator = Matrix{ComplexF64})      # dense reference

- `max_dt`: largest internal QP step (Inf = one step per output interval; optimal for Cheby).
- `block = true`: propagate the N×p block as one state.
Negative times use `backward = true`; mixed signs are allowed.
"""
struct QPAlg{M,KW<:NamedTuple} <: AbstractStatePropagatorAlg
    method::M
    max_dt::Float64
    block::Bool
    kwargs::KW
end
QPAlg(method = QP.Cheby; max_dt::Real = Inf, block::Bool = false, kwargs...) =
    QPAlg(method, Float64(max_dt), block, (; kwargs...))

_qp_name(m::Module) = Symbol(:qp_, nameof(m))
_qp_name(m) = Symbol(:qp_, m)

"Chebychev: estimate the spectral range once (not in every init_prop), unless given."
function _init_prop_kwargs(H, alg::QPAlg)
    kw = alg.kwargs
    is_cheby = alg.method === QP.Cheby || alg.method === :Cheby
    (is_cheby && !haskey(kw, :specrange_method)) || return kw
    E_min, E_max = QP.SpectralRange.specrange(H, :auto)
    return merge(kw, (; specrange_method = :manual, E_min, E_max))
end

"""
    _qp_sweeps(tsv)
QPAlg's schedule: per direction (t ≥ 0 forward, t < 0 backward) the output indices in order
of |t| and the step δ ≥ 0 leading to each (0 for zero / repeated times). Shared with the
cost model so that both count exactly the same steps.
"""
function _qp_sweeps(tsv)
    map((false, true)) do backward
        idx = findall(t -> backward ? t < 0 : t ≥ 0, tsv)
        idx = idx[sortperm(abs.(tsv[idx]))]
        τ = Float64.(abs.(tsv[idx]))
        δs = diff([0.0; τ])
        δs[δs .≤ 1e-12 .* τ] .= 0.0
        (; backward, idx, δs)
    end
end
_step_key(δ) = round(δ; sigdigits = 12)

function propagate_block(H, Ψ0, ts, alg::QPAlg)
    Hop = _operator(H)
    tsv = _tvec(ts)
    N, p = size(Ψ0)
    Us = [Matrix{ComplexF64}(undef, N, p) for _ in tsv]
    kw = any(!iszero, tsv) ? _init_prop_kwargs(Hop, alg) : alg.kwargs
    props = Dict{Tuple{Bool,Float64},Any}()          # one QP propagator per (direction, δ)
    nsteps = 0
    for cols in (alg.block ? (Colon(),) : 1:p), sw in _qp_sweeps(tsv)
        ψ = alg.block ? Matrix{ComplexF64}(Ψ0) : Vector{ComplexF64}(Ψ0[:, cols])
        for (j, δ) in zip(sw.idx, sw.δs)
            if δ > 0
                prop = get!(props, (sw.backward, _step_key(δ))) do
                    nsub = isfinite(alg.max_dt) ? max(1, ceil(Int, δ / alg.max_dt)) : 1
                    QP.init_prop(ψ, Hop, collect(range(0.0, δ; length = nsub + 1));
                        method = alg.method, backward = sw.backward, inplace = true, kw...)
                end
                QP.reinit_prop!(prop, ψ)
                while QP.prop_step!(prop) !== nothing
                    nsteps += 1
                end
                copyto!(ψ, prop.state)
            end
            Us[j][:, cols] .= ψ
        end
    end
    return Us, (; method = _qp_name(alg.method), err_est = NaN, nmatvec = -1,
                  nsteps, nprops = length(props))
end

# -----------------------------------------------------------------------------
# 2. Automatic selection: Diagonalization vs QP.Cheby
# -----------------------------------------------------------------------------
"""
    AutoPropagatorAlg(; tol = 1e-10, candidates = (:diag, :cheby), kwargs...)

Picks the cheapest estimated execution time between:
  :diag    DiagonalizationPropagatorAlg   f·sec_eig·N³ + sec_gemm·N²·p·(1 + n_t)
  :cheby   QPAlg(QP.Cheby)                Σ_steps K(Δ, δ) · p · (matvec + 4 vec)
where K(Δ, δ) = length(QP.Cheby.cheby_coeffs(Δ, δ; limit = tol)) − 1 is QP's Chebychev term count,
Δ = E_max − E_min is QP's spectral range, and the steps δ are the ones taken by `_qp_sweeps`.

- `small_dim`: diagonalize without further analysis (if :diag is a candidate).
- `max_diag_dim`: never diagonalize above this dimension (memory/scaling ceiling).
- The spectral range is only estimated if diagonalization is not already cheaper than the
  estimate itself (≈ `specrange_matvecs` SpMVs); it is then reused directly by Chebychev.
"""
Base.@kwdef struct AutoPropagatorAlg <: AbstractStatePropagatorAlg
    tol::Float64 = 1e-10
    candidates::Tuple{Vararg{Symbol}} = (:diag, :cheby)
    small_dim::Int = 200
    max_diag_dim::Int = 4000
    block::Bool = false                   # forwarded to QPAlg(QP.Cheby)
    specrange_matvecs::Int = 60
    specrange_kwargs::NamedTuple = (;)    # forwarded to QP.SpectralRange.specrange
    sec_eig::Float64 = 4e-10              # per N³ (complex Hermitian eigen)
    real_eig_factor::Float64 = 0.35
    sec_gemm::Float64 = 1.6e-10           # per N²·column
    sec_nnz::Float64 = 1.3e-9             # per stored entry of H per column (matvec)
    sec_vec::Float64 = 6.6e-10            # per vector element op
end

_matvec_cost(H::SparseMatrixCSC, a) = a.sec_nnz * nnz(H)
_matvec_cost(H::AbstractMatrix, a) = a.sec_nnz * length(H)
_matvec_cost(H, a) = a.sec_nnz * float(size(H, 1))^2

function _cost_diag(H, N, p, nt, a)
    f = eltype(H) <: Real ? a.real_eig_factor : 1.0
    return f * a.sec_eig * float(N)^3 + a.sec_gemm * float(N)^2 * p * (1 + nt)
end

function _step_multiplicities(tsv)
    steps = Dict{Float64,Int}()
    for sw in _qp_sweeps(tsv), δ in sw.δs
        δ > 0 && (steps[_step_key(δ)] = get(steps, _step_key(δ), 0) + 1)
    end
    return steps
end

_ncheb(Δ, δ, tol) = length(QP.Cheby.cheby_coeffs(Δ, δ; limit = tol)) - 1

function select_propagator(H, Ψ0, ts, a::AutoPropagatorAlg)
    Hop = _operator(H)
    tsv = _tvec(ts)
    N, p = size(Ψ0)
    nt = length(tsv)
    want(c) = c in a.candidates
    rep = (; choice = :none, reason = :none, cost_diag = NaN, cost_cheby = NaN,
             Δ = NaN, ρT = NaN)
    diag = DiagonalizationPropagatorAlg()

    all(iszero, tsv) && return QPAlg(), merge(rep, (; reason = :no_propagation))
    can_diag = want(:diag) && N ≤ a.max_diag_dim
    can_cheby = want(:cheby)
    cost_diag = can_diag ? _cost_diag(Hop, N, p, nt, a) : Inf
    rep = merge(rep, (; cost_diag))

    if can_diag && (N ≤ a.small_dim || !can_cheby)
        return diag, merge(rep, (; choice = :diag, reason = :small))
    end
    can_cheby || throw(ArgumentError("AutoPropagatorAlg: no admissible candidate for N = $N"))

    mv = _matvec_cost(Hop, a)
    vop = a.sec_vec * N
    m_sr = a.specrange_matvecs
    if cost_diag ≤ m_sr * (mv + m_sr / 2 * vop)
        return diag, merge(rep, (; choice = :diag, reason = :cheaper_than_specrange))
    end

    E_min, E_max = QP.SpectralRange.specrange(Hop, :auto; a.specrange_kwargs...)
    Δ = max(E_max - E_min, eps() * max(1.0, abs(E_max)))
    steps = _step_multiplicities(tsv)
    T = maximum(abs, tsv)

    cost_cheby = p * (sum(n * _ncheb(Δ, δ, a.tol) for (δ, n) in steps) * (mv + 4vop) + nt * vop)

    choice = cost_diag ≤ cost_cheby ? :diag : :cheby
    chosen = choice === :diag ? diag :
             QPAlg(QP.Cheby; block = a.block, cheby_coeffs_limit = a.tol,
                   specrange_method = :manual, E_min, E_max)

    return chosen, merge(rep, (; choice, reason = :cost, cost_cheby, Δ, ρT = (Δ / 2) * T))
end

function propagate_block(H, Ψ0, ts, alg::AutoPropagatorAlg)
    chosen, selection = select_propagator(H, Ψ0, ts, alg)
    Us, info = propagate_block(H, Ψ0, ts, chosen)
    return Us, merge(info, (; selection))
end
