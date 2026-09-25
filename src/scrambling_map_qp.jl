# =============================================================================
# scrambling_map_qp.jl — QuantumPropagators.jl backend.
# Include AFTER scrambling_map.jl. Enclosing module needs:
#     using QuantumPropagators      # init_prop, prop_step!, reinit_prop!
#     using ExponentialUtilities    # activates QP's ExponentialUtilities extension
# Uses from scrambling_map.jl: AbstractStatePropagatorAlg, _operator, _tvec,
# lanczos_bounds, gershgorin_bounds.
# =============================================================================

import QuantumPropagators
const QP = QuantumPropagators

abstract type AbstractQPSubAlg end

"""
    QPCheby(; cheby_coeffs_limit = 1e-12, bounds = :lanczos, specrange_buffer = 0.01,
              lanczos_steps = 30, check_normalization = false, max_ρdt = Inf)

QuantumPropagators' Chebychev propagator (column-wise, or whole block with block = true).
- `cheby_coeffs_limit`: coefficients below this are dropped ≈ per-step accuracy.
- `bounds`: `:lanczos` (our estimate ∩ Gershgorin; no RNG side effects), `:gershgorin`
  (guaranteed), `:qp` (QP's own specrange), or `(E_min, E_max)`.
- `specrange_buffer`: QP enlarges the range by this fraction of its width.
- `check_normalization`: QP checks for an underestimated spectral range (debugging).
  Unlike ChebyshevPropagatorAlg there is no automatic restart.
- `max_ρdt`: max ρ·dt per QP step (Inf = one step per output interval, optimal for Chebyshev).
"""
Base.@kwdef struct QPCheby <: AbstractQPSubAlg
    cheby_coeffs_limit::Float64 = 1e-12
    bounds::Any = :lanczos
    specrange_buffer::Float64 = 0.01
    lanczos_steps::Int = 30
    check_normalization::Bool = false
    max_ρdt::Float64 = Inf
end

"""
    QPNewton(; m_max = 10, relerr = 1e-12, max_restarts = 50, max_ρdt = 20.0)
Restarted Newton/Leja Krylov propagator. Keep ρ·dt well below m_max·max_restarts.
"""
Base.@kwdef struct QPNewton <: AbstractQPSubAlg
    m_max::Int = 10
    relerr::Float64 = 1e-12
    max_restarts::Int = 50
    max_ρdt::Float64 = 20.0
end

"""
    QPExpUtils(; m = 30, tol = 1e-10, mode = :error_estimate, iop = 0, max_ρdt = 12.0)
QP wrapper around ExponentialUtilities (time argument -i·dt on the Hermitian H → Lanczos).
With `mode = :error_estimate`, `tol` is a per-step accuracy target; with `:happy_breakdown`
it is only the breakdown threshold (no accuracy control). Keep max_ρdt ≲ 0.4·m.
"""
Base.@kwdef struct QPExpUtils <: AbstractQPSubAlg
    m::Int = 30
    tol::Float64 = 1e-10
    mode::Symbol = :error_estimate
    iop::Int = 0
    max_ρdt::Float64 = 12.0
end

"""
    QPExpProp(; max_ρdt = Inf)
Dense matrix exponential per step. Reference / small-N only.
"""
Base.@kwdef struct QPExpProp <: AbstractQPSubAlg
    max_ρdt::Float64 = Inf
end

"""
    QPAlg(sub = QPCheby(); block = false, reuse = true)
Propagation via QuantumPropagators.jl with sub-algorithm `sub`.
- `block = true`: propagate the N×p matrix as one state (experimental; not for QPExpUtils).
- `reuse = true`: reuse each propagator across columns via `reinit_prop!`.
"""
struct QPAlg{S <: AbstractQPSubAlg} <: AbstractStatePropagatorAlg
    sub::S
    block::Bool
    reuse::Bool
end
function QPAlg(sub::AbstractQPSubAlg = QPCheby(); block::Bool = false, reuse::Bool = true)
    QPAlg{typeof(sub)}(sub, block, reuse)
end

_qp_method(::QPCheby) = QP.Cheby
_qp_method(::QPNewton) = QP.Newton
_qp_method(::QPExpProp) = QP.ExpProp
_qp_method(::QPExpUtils) = ExponentialUtilities

_qp_name(::QPCheby) = :qp_cheby
_qp_name(::QPNewton) = :qp_newton
_qp_name(::QPExpProp) = :qp_expprop
_qp_name(::QPExpUtils) = :qp_exponentialutilities

function _qp_kwargs(s::QPCheby, bnds, N)
    kw = (; cheby_coeffs_limit = s.cheby_coeffs_limit,
        specrange_buffer = s.specrange_buffer,
        check_normalization = s.check_normalization)
    (s.bounds === :qp || bnds === nothing) && return kw
    return merge(kw, (; specrange_method = :manual, E_min = bnds[1], E_max = bnds[2]))
end
_qp_kwargs(s::QPNewton, _, N) = (; m_max = s.m_max, relerr = s.relerr, max_restarts = s.max_restarts)
function _qp_kwargs(s::QPExpUtils, _, N)
    (; expv_kwargs = (; m = min(s.m, N), tol = s.tol, mode = s.mode, iop = s.iop, ishermitian = true))
end
_qp_kwargs(::QPExpProp, _, N) = (; convert_operator = Matrix{ComplexF64})

_needs_bounds(s::QPCheby) = s.bounds !== :qp || isfinite(s.max_ρdt)
_needs_bounds(s::AbstractQPSubAlg) = isfinite(s.max_ρdt)

"Spectral interval (lo, hi) of H for Chebyshev scaling and substep sizing."
function _qp_bounds(H, s::AbstractQPSubAlg)
    b = s isa QPCheby ? s.bounds : :lanczos
    steps = s isa QPCheby ? s.lanczos_steps : 30
    g = gershgorin_bounds(H)
    if b === :gershgorin
        g === nothing && throw(ArgumentError("Gershgorin bounds need an explicit matrix H"))
        return g
    elseif b isa Tuple
        return (Float64(b[1]), Float64(b[2]))
    else                                    # :lanczos, or :qp (ρ only, for substeps)
        lo, hi = lanczos_bounds(H; steps)
        if g !== nothing
            lo = max(lo, g[1])
            hi = min(hi, g[2])
        end
        return (lo, hi)
    end
end

"""
    _qp_runs(τs, ρ, max_ρdt; rtol = 1e-10)
Split sorted non-negative times into runs of equal spacing δ (starting from 0). Each run
gets a uniform QP grid with `nsub` steps per output interval (ρ·δ/nsub ≤ max_ρdt).
"""
function _qp_runs(τs::AbstractVector{<:Real}, ρ::Real, max_ρdt::Real; rtol = 1e-10)
    runs = @NamedTuple{δ::Float64, count::Int, nsub::Int, tlist::Vector{Float64}}[]
    n = length(τs)
    n == 0 && return runs
    scale = max(abs(float(τs[end])), floatmin(Float64))
    clean(d) = d ≤ rtol * scale ? 0.0 : Float64(d)
    prev = 0.0
    i = 1
    while i ≤ n
        δ = clean(τs[i] - prev)
        cnt = 1
        while i + cnt ≤ n && abs(clean(τs[i + cnt] - τs[i + cnt - 1]) - δ) ≤ rtol * scale
            cnt += 1
        end
        if iszero(δ)
            push!(runs, (; δ, count = cnt, nsub = 0, tlist = Float64[]))
        else
            nsub = (isfinite(ρ) && isfinite(max_ρdt)) ? max(1, ceil(Int, ρ * δ / max_ρdt)) : 1
            tl = collect(range(0.0, cnt * δ; length = cnt * nsub + 1))
            push!(runs, (; δ, count = cnt, nsub, tlist = tl))
        end
        prev = Float64(τs[i + cnt - 1])
        i += cnt
    end
    return runs
end

function propagate_block(H, Ψ0, ts, alg::QPAlg)
    sub = alg.sub
    if alg.block && sub isa QPExpUtils
        throw(ArgumentError("QPExpUtils needs vector states (ExponentialUtilities Krylov); use block = false"))
    end
    Hop = _operator(H)
    tsv = _tvec(ts)
    N, p = size(Ψ0)
    Us = [Matrix{ComplexF64}(undef, N, p) for _ in tsv]
    isempty(tsv) && return Us, (; method = _qp_name(sub), err_est = 0.0, nmatvec = 0, nsteps = 0, ρ = NaN)
    s = all(≥(0), tsv) ? 1 : all(≤(0), tsv) ? -1 :
        throw(ArgumentError("QPAlg: all times must have the same sign"))
    G = s == 1 ? Hop : -Hop                        # exp(-iH t) = exp(-i(-H)|t|) for t < 0
    bnds = nothing
    ρ = NaN
    if _needs_bounds(sub)
        lo, hi = _qp_bounds(Hop, sub)
        bnds = s == 1 ? (lo, hi) : (-hi, -lo)
        ρ = (hi - lo) / 2
    end
    τ = abs.(tsv)
    perm = sortperm(τ)
    runs = _qp_runs(τ[perm], ρ, sub.max_ρdt)
    method = _qp_method(sub)
    kw = _qp_kwargs(sub, bnds, N)
    props = Vector{Any}(nothing, length(runs))
    colsets = alg.block ? Any[Colon()] : Any[n for n in 1:p]
    nsteps = 0
    for cols in colsets
        ψ = alg.block ? Matrix{ComplexF64}(Ψ0) : Vector{ComplexF64}(Ψ0[:, cols])
        j = 0
        for (r, run) in enumerate(runs)
            if run.nsub == 0                           # duplicate / zero times
                for _ in 1:run.count
                    j += 1
                    Us[perm[j]][:, cols] .= ψ
                end
                continue
            end
            prop = props[r]
            if prop === nothing || !alg.reuse
                prop = QP.init_prop(copy(ψ), G, run.tlist; method, inplace = true, kw...)
                props[r] = prop
            else
                QP.reinit_prop!(prop, copy(ψ))
            end
            for _ in 1:run.count
                for _ in 1:run.nsub
                    QP.prop_step!(prop)
                end
                nsteps += run.nsub
                j += 1
                Us[perm[j]][:, cols] .= prop.state
            end
            ψ = copy(prop.state)
        end
    end
    return Us, (; method = _qp_name(sub), err_est = NaN, nmatvec = -1, nsteps, ρ)
end