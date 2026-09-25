# =============================================================================
# scrambling_map.jl
#
# Enclosing module (QDELM) must have:
#     using LinearAlgebra, SparseArrays, Random
#     using ExponentialUtilities
# and QDELM's own: dim, generalized_kron, density_matrix, to_dense, propagator,
# operator_time_evolution, effective_measurement, QuantumDotSystem.
#
# Central primitive:
#     Us, info = propagate_block(H, Ψ0, ts, alg)
# Us[j] ≈ exp(-i H ts[j]) Ψ0 for an N×p block Ψ0.
# info::NamedTuple always contains (method, err_est, nmatvec).
# =============================================================================

abstract type AbstractPropagatorAlg end
"Algorithms that propagate the pure-state block Ψ0 = [e_j ⊗ ψres]_j."
abstract type AbstractStatePropagatorAlg <: AbstractPropagatorAlg end

# -----------------------------------------------------------------------------
# 1. Operator path (unchanged)
# -----------------------------------------------------------------------------
struct BlockPropagatorAlg <: AbstractPropagatorAlg end

function scrambling_map(H_main, H_res, H_total, measurements, ψres, hamiltonian, t::Number, ::BlockPropagatorAlg)
    ρ_res = density_matrix(ψres)
    U = propagator(t, hamiltonian)
    measurements_t = map(Base.Fix1(operator_time_evolution, U), measurements)
    eff_measurements = map(
        mt -> effective_measurement(mt, ρ_res, H_main, H_res, H_total),
        measurements_t)
    return reduce(vcat, (vec(m)' for m in eff_measurements))
end

function scrambling_map(H_main, H_res, H_total, measurements, ψres, hamiltonian, ts::AbstractArray, alg::BlockPropagatorAlg)
    mapreduce(t -> scrambling_map(H_main, H_res, H_total, measurements, ψres,
        hamiltonian, t, alg), vcat, ts)
end

# -----------------------------------------------------------------------------
# 2. Common helpers and scrambling_map front end
# -----------------------------------------------------------------------------
_tvec(t::Real) = [t]
_tvec(ts) = collect(ts)
_increments(ts::Vector{T}) where T = isempty(ts) ? T[] : [ts[1]; diff(ts)]

_operator(H::SparseMatrixCSC) = H
_operator(H::StridedMatrix{<:Complex}) = H
_operator(H::StridedMatrix{R}) where R<:Real = Matrix{complex(R)}(H)   # keep BLAS on complex blocks
_operator(H::AbstractMatrix) = sparse(H)
_operator(H) = H

_nnz_per_row(H::SparseMatrixCSC) = nnz(H) / size(H, 1)
_nnz_per_row(H) = float(size(H, 2))

"""
    initial_block(H_main, H_res, H_total, ψres) -> Ψ0 :: Matrix{ComplexF64} (N × N_main)
Columns are e_j ⊗ ψres embedded in the total Hilbert space.
"""
function initial_block(H_main, H_res, H_total, ψres::AbstractVector{T}) where T
    N_main = dim(H_main)
    e_j = zeros(T, N_main)
    stack(1:N_main) do n
        fill!(e_j, 0)
        e_j[n] = 1
        generalized_kron((e_j, ψres), (H_main, H_res) => H_total)
    end
end

"Rows vec(U' D_k U)' for each (diagonal) measurement D_k — same convention as before."
measurement_rows(U, measurements) = stack(op -> vec(U' * (Diagonal(op) * U)), measurements)'

function scrambling_map(H_main, H_res, H_total, measurements, ψres::AbstractVector,
    hamiltonian, t::Number, alg::AbstractStatePropagatorAlg)
    Ψ0 = initial_block(H_main, H_res, H_total, ψres)
    Us, _ = propagate_block(hamiltonian, Ψ0, _tvec(t), alg)
    return measurement_rows(only(Us), measurements)
end

function scrambling_map(H_main, H_res, H_total, measurements, ψres::AbstractVector,
    hamiltonian, ts::AbstractVector{<:Real}, alg::AbstractStatePropagatorAlg)
    Ψ0 = initial_block(H_main, H_res, H_total, ψres)
    Us, _ = propagate_block(hamiltonian, Ψ0, ts, alg)
    return reduce(vcat, [measurement_rows(U, measurements) for U in Us])
end

# -----------------------------------------------------------------------------
# 3. Diagonalization (one eigen for all times, BLAS-3 block products)
# -----------------------------------------------------------------------------
"""
    DiagonalizationPropagatorAlg()
Dense Hermitian eigendecomposition once; every time point is then two GEMMs.
"""
struct DiagonalizationPropagatorAlg <: AbstractStatePropagatorAlg end

function propagate_block(H, Ψ0, ts, ::DiagonalizationPropagatorAlg)
    tsv = _tvec(ts)
    F = eigen(Hermitian(to_dense(H)))
    C = F.vectors' * Ψ0
    tmp = similar(C, complex(eltype(C)))
    Us = map(tsv) do t
        tmp .= cis.(-t .* F.values) .* C
        F.vectors * tmp
    end
    tmax = isempty(tsv) ? 0.0 : maximum(abs, tsv)
    err = eps() * size(H, 1) * maximum(abs, F.values) * max(1.0, tmax)   # heuristic
    return Us, (; method=:diagonalization, err_est=err, nmatvec=0)
end

# -----------------------------------------------------------------------------
# 4. Spectral bounds
# -----------------------------------------------------------------------------
"""
    gershgorin_bounds(H) -> (λlo, λhi) or nothing
Guaranteed enclosure of the spectrum, O(nnz).
"""
function gershgorin_bounds(H::SparseMatrixCSC)
    rows, vals = rowvals(H), nonzeros(H)
    lo, hi = Inf, -Inf
    for j in axes(H, 2)
        d = 0.0
        rad = 0.0
        for q in nzrange(H, j)
            i = rows[q]
            if i == j
                d = real(vals[q])
            else
                rad += abs(vals[q])
            end
        end
        lo = min(lo, d - rad)
        hi = max(hi, d + rad)
    end
    pad = 1e-12 * max(1.0, abs(lo), abs(hi))
    return (lo - pad, hi + pad)
end
function gershgorin_bounds(H::StridedMatrix)
    lo, hi = Inf, -Inf
    for j in axes(H, 2)
        d = real(H[j, j])
        rad = 0.0
        for i in axes(H, 1)
            i == j || (rad += abs(H[i, j]))
        end
        lo = min(lo, d - rad)
        hi = max(hi, d + rad)
    end
    pad = 1e-12 * max(1.0, abs(lo), abs(hi))
    return (lo - pad, hi + pad)
end
gershgorin_bounds(H) = nothing

"""
    lanczos_bounds(H; steps = 30, seed = 0x5eed) -> (λlo, λhi)
Zhou-Li style estimate θ_min - β_m, θ_max + β_m from a short fully re-orthogonalized
Lanczos run. Uses its own RNG (does not consume the task-local stream).
Not guaranteed — Chebyshev monitors it and clamps to Gershgorin.
"""
function lanczos_bounds(H; steps::Int=30, seed=0x5eed)
    N = size(H, 1)
    k = min(steps, N)
    rng = Random.Xoshiro(seed)
    T = complex(eltype(H))
    V = Matrix{T}(undef, N, k + 1)
    v = randn(rng, T, N)
    V[:, 1] .= v ./ norm(v)
    α = zeros(k)
    β = zeros(k)
    w = Vector{T}(undef, N)
    m = k
    for j in 1:k
        vj = view(V, :, j)
        mul!(w, H, vj)
        α[j] = real(dot(vj, w))
        Vj = view(V, :, 1:j)
        for _ in 1:2                                  # full re-orthogonalization
            h = Vj' * w
            mul!(w, Vj, h, -1, 1)
        end
        β[j] = norm(w)
        if β[j] ≤ 1e-12 * max(1.0, abs(α[j]))         # invariant subspace
            m = j
            break
        end
        j < k && (V[:, j+1] .= w ./ β[j])
    end
    θ = eigvals(SymTridiagonal(α[1:m], β[1:(m-1)]))
    return (θ[1] - β[m], θ[end] + β[m])
end

# -----------------------------------------------------------------------------
# 5. Chebyshev
# -----------------------------------------------------------------------------
"""
    bessel_j_sequence(x) -> J with J[k+1] ≈ J_k(x), k = 0..M, x ≥ 0
Miller backward recurrence normalized by J_0 + 2Σ J_{2k} = 1. Entries beyond M are
negligible (< 1e-20). No SpecialFunctions dependency.
"""
function bessel_j_sequence(x::T) where T
    # x = Float64(x)
    x < 0 && throw(DomainError(x, "x must be ≥ 0"))
    x == 0 && return [one(T)]
    x < 1e-6 && return [1 - x^2 / 4, x / 2 - x^3 / 16, x^2 / 8]
    M = ceil(Int, x + 20 * cbrt(x) + 40)
    while true
        J = _miller_bessel(x, M)
        maximum(abs, @view J[(end-9):end]) < 1e-20 && return J
        M = ceil(Int, 1.5 * M)
    end
end

function _miller_bessel(x::T, M::Int) where T
    invx = 1 / x
    J = zeros(typeof(invx), M + 1)                 # J[k+1] ↔ order k
    jk1 = 0.0
    jk = 1.0
    J[M+1] = jk
    for k in M:-1:1
        jkm1 = (2k * invx) * jk - jk1
        J[k] = jkm1
        jk1, jk = jk, jkm1
        if abs(jk) > 1e200           # rescale to avoid overflow
            @views J[k:end] .*= 1e-200
            jk *= 1e-200
            jk1 *= 1e-200
        end
    end
    S = J[1] + 2 * sum(@view J[3:2:end])
    J ./= S
    return J
end

"""
    chebyshev_coefficients(x, tol) -> (a, tail)
exp(-i x y) = Σ_{k=0}^{K} a_k T_k(y) + R, |R| ≤ tail ≤ tol for y ∈ [-1, 1].
"""
function chebyshev_coefficients(x::Real, tol::Real)
    J = bessel_j_sequence(abs(x))
    M = length(J) - 1
    acc = 0.0
    K = 0
    for k in M:-1:1
        a = acc + 2 * abs(J[k+1])
        if a > tol
            K = k
            break
        end
        acc = a
    end
    ζ = x ≥ 0 ? -im : im
    coeffs = [(k == 0 ? 1.0 : 2.0) * ζ^mod(k, 4) * J[k+1] for k in 0:K]
    return coeffs, acc
end

"""
    ChebyshevPropagatorAlg(; tol = 1e-10, bounds = :lanczos, margin = 0.01,
                             lanczos_steps = 30, multitime = :auto, growth_limit = 1.5)

Block Chebyshev (Tal-Ezer-Kosloff) propagation of the whole N×p block.
- `tol`: bound on the 2-norm error per unit-norm column at every requested time.
- `bounds`: `:lanczos` (estimate + margin, clamped to Gershgorin), `:gershgorin`
  (guaranteed), or a tuple `(λmin, λmax)` (treated as an estimate).
- `multitime`: `:onepass` (one recurrence, per-time coefficients), `:stepping`
  (Δt jumps), or `:auto` (cost-based).
- `growth_limit`: if max_k ‖T_k Ψ0‖/‖Ψ0‖ exceeds this, the bounds are declared
  violated and the run restarts with guaranteed Gershgorin bounds.
"""
Base.@kwdef struct ChebyshevPropagatorAlg <: AbstractStatePropagatorAlg
    tol::Float64 = 1e-10
    bounds::Any = :lanczos
    margin::Float64 = 0.01
    lanczos_steps::Int = 30
    multitime::Symbol = :auto
    growth_limit::Float64 = 1.5
end

"Returns (c, r, guaranteed) with spec(H) ⊂ [c − r, c + r] (guaranteed or estimated)."
function chebyshev_interval(H, alg::ChebyshevPropagatorAlg)
    g = gershgorin_bounds(H)
    b = alg.bounds
    if b === :gershgorin
        g === nothing && throw(ArgumentError("Gershgorin bounds need an explicit matrix H"))
        lo, hi = g
        guaranteed = true
    else
        lo, hi = if b === :lanczos
            lanczos_bounds(H; steps=alg.lanczos_steps)
        elseif b isa Tuple
            (Float64(b[1]), Float64(b[2]))
        else
            throw(ArgumentError("unknown bounds option $b"))
        end
        pad = alg.margin * (hi - lo) / 2 + 1e-12 * max(1.0, abs(lo), abs(hi))
        lo -= pad
        hi += pad
        guaranteed = false
        if g !== nothing                 # never wider than a guaranteed interval
            lo = max(lo, g[1])
            hi = min(hi, g[2])
            guaranteed = (lo == g[1]) && (hi == g[2])
        end
    end
    c = (lo + hi) / 2
    r = max((hi - lo) / 2, 1e-12 * max(1.0, abs(c)))
    return c, r, guaranteed
end

function _step_coefficients(r, tsv, tol)
    δs = _increments(tsv)
    nsteps = max(count(!iszero, δs), 1)
    return [iszero(δ) ? nothing : chebyshev_coefficients(r * δ, tol / nsteps) for δ in δs]
end

"Plan: coefficient sets for the chosen mode and its work (matvecs, block axpys)."
function _chebyshev_plan(tsv, r, alg::ChebyshevPropagatorAlg, w)
    n = length(tsv)
    mode = alg.multitime
    mode in (:auto, :onepass, :stepping) || throw(ArgumentError("multitime = $mode"))
    (mode === :auto && n == 1) && (mode = :onepass)
    one = mode === :stepping ? nothing : [chebyshev_coefficients(r * t, alg.tol) for t in tsv]
    step = mode === :onepass ? nothing : _step_coefficients(r, tsv, alg.tol)
    work_one = one === nothing ? (Inf, Inf) :
               (maximum(c -> length(c[1]) - 1, one), sum(c -> length(c[1]), one))
    work_step = step === nothing ? (Inf, Inf) : begin
        K = sum(s -> s === nothing ? 0 : length(s[1]) - 1, step)
        (K, K + n)
    end
    cost(wk) = wk[1] * (w + 3) + wk[2]            # units of N·p complex FMAs
    if mode === :auto
        mode = cost(work_one) ≤ cost(work_step) ? :onepass : :stepping
    end
    matvecs, axpys = mode === :onepass ? work_one : work_step
    return (; mode, one, step, matvecs, axpys)
end

"Core recurrence. Returns (outs, stats) or `nothing` if spectral bounds are violated."
function _cheb_kernel(H, Ψ0::AbstractMatrix, coeffs::Vector{Vector{T}},
    c, r, growth_limit) where T
    K = maximum(length, coeffs) - 1
    outs = [Matrix{T}(a[1] .* Ψ0) for a in coeffs]
    K == 0 && return outs, (; nmatvec=0, growth=1.0)
    nrm0 = norm(Ψ0)
    Tprev = Matrix{T}(Ψ0)                 # T_0 Ψ0
    Tcur = similar(Tprev)
    mul!(Tcur, H, Tprev)
    @. Tcur = (Tcur - c * Tprev) / r               # T_1 Ψ0 = H_s Ψ0
    for (U, a) in zip(outs, coeffs)
        length(a) ≥ 2 && axpy!(a[2], Tcur, U)
    end
    growth = norm(Tcur) / nrm0
    growth > growth_limit && return nothing
    for k in 2:K
        mul!(Tprev, H, Tcur, 2 / r, -1.0)          # (2/r) H T_{k-1} − T_{k-2}
        axpy!(-2c / r, Tcur, Tprev)                # … − (2c/r) T_{k-1}
        Tprev, Tcur = Tcur, Tprev                  # Tcur = T_k Ψ0
        for (U, a) in zip(outs, coeffs)
            length(a) > k && axpy!(a[k+1], Tcur, U)
        end
        if k ≤ 8 || k % 8 == 0 || k == K
            gk = norm(Tcur) / nrm0
            growth = max(growth, gk)
            gk > growth_limit && return nothing
        end
    end
    return outs, (; nmatvec=K, growth)
end

function _chebyshev_run(H, Ψ0, tsv, c, r, alg, w)
    plan = _chebyshev_plan(tsv, r, alg, w)
    if plan.mode === :onepass
        res = _cheb_kernel(H, Ψ0, [cf[1] for cf in plan.one], c, r, alg.growth_limit)
        res === nothing && return nothing
        outs, st = res
        for (U, t) in zip(outs, tsv)
            U .*= cis(-c * t)
        end
        tail = isempty(plan.one) ? 0.0 : maximum(cf -> cf[2], plan.one)
        err = st.growth * tail + st.nmatvec * eps()
        return outs, (; method=:chebyshev, mode=:onepass, err_est=err,
            nmatvec=st.nmatvec, growth=st.growth, c, r)
    else
        cur = Matrix{ComplexF64}(Ψ0)
        outs = Vector{Matrix{ComplexF64}}(undef, length(tsv))
        nmv = 0
        err = 0.0
        g = 1.0
        for (i, δ) in enumerate(_increments(tsv))
            s = plan.step[i]
            if s !== nothing
                res = _cheb_kernel(H, cur, [s[1]], c, r, alg.growth_limit)
                res === nothing && return nothing
                o, st = res
                cur = o[1]
                cur .*= cis(-c * δ)
                nmv += st.nmatvec
                err += st.growth * s[2] + st.nmatvec * eps()
                g = max(g, st.growth)
            end
            outs[i] = copy(cur)
        end
        return outs, (; method=:chebyshev, mode=:stepping, err_est=err,
            nmatvec=nmv, growth=g, c, r)
    end
end

function propagate_block(H, Ψ0, ts, alg::ChebyshevPropagatorAlg)
    Hop = _operator(H)
    tsv = _tvec(ts)
    w = _nnz_per_row(Hop)
    c, r, guaranteed = chebyshev_interval(Hop, alg)
    res = _chebyshev_run(Hop, Ψ0, tsv, c, r, alg, w)
    restarted = false
    tries = 0
    while res === nothing
        guaranteed && error("Chebyshev diverged with guaranteed (Gershgorin) bounds — is H Hermitian?")
        (tries += 1) > 4 && error("Chebyshev: could not establish valid spectral bounds")
        g = gershgorin_bounds(Hop)
        if g === nothing
            r *= 2                                  # no guaranteed bound available
        else
            c, r, guaranteed = (g[1] + g[2]) / 2, (g[2] - g[1]) / 2, true
        end
        restarted = true
        res = _chebyshev_run(Hop, Ψ0, tsv, c, r, alg, w)
    end
    Us, info = res
    return Us, merge(info, (; restarted, guaranteed))
end

# -----------------------------------------------------------------------------
# 7a. ExponentialUtilities: lanczos! on H (complex time) + a-posteriori step control
# -----------------------------------------------------------------------------
"""
    ExpUtilsLanczosPropagatorAlg(; m = 30, tol = 1e-10, breakdown_tol = 1e-13, max_substeps = 100_000)
Lanczos basis from `ExponentialUtilities.lanczos!` on the Hermitian H (short recurrence,
no O(m²N) Arnoldi orthogonalization). Each basis gives the largest dt with Saad's estimate
β·h_{m+1,m}·|e_mᵀ exp(-i dt T_m) e_1| ≤ tol_rate·dt; rejections cost no matvecs.
Uses KrylovSubspace internals (m, beta, H, V).
"""
Base.@kwdef struct ExpUtilsLanczosPropagatorAlg <: AbstractStatePropagatorAlg
    m::Int = 30
    tol::Float64 = 1e-10
    breakdown_tol::Float64 = 1e-13
    max_substeps::Int = 100_000
end

function _eu_lanczos_step!(ψ, H, δ, Ks, m, tol_rate, alg)
    iszero(δ) && return 0.0, 0
    remaining = δ
    err = 0.0
    nmv = 0
    nsub = 0
    while !iszero(remaining)
        (nsub += 1) > alg.max_substeps && error("ExpUtilsLanczosPropagatorAlg: too many substeps")
        ExponentialUtilities.lanczos!(Ks, H, ψ; m=m, tol=alg.breakdown_tol)
        mk = Ks.m
        β = Ks.beta
        nmv += mk
        α = [real(Ks.H[i, i]) for i in 1:mk]
        b = [real(Ks.H[i+1, i]) for i in 1:(mk-1)]
        hnext = abs(Ks.H[mk+1, mk])
        F = eigen(SymTridiagonal(α, b))
        v1 = F.vectors[1, :]
        vm = F.vectors[mk, :]
        est(dt) = β * hnext * abs(sum(vm .* cis.(-dt .* F.values) .* v1))
        dt = remaining
        e = est(dt)
        if mk == m                        # no happy breakdown → control the step
            for _ in 1:100
                e ≤ tol_rate * abs(dt) && break
                dt *= clamp(0.9 * (tol_rate * abs(dt) / e)^(1 / max(mk - 1, 1)), 0.05, 0.9)
                e = est(dt)
            end
            e ≤ tol_rate * abs(dt) || error("ExpUtilsLanczosPropagatorAlg: step-size control failed")
        end
        y = F.vectors * (cis.(-dt .* F.values) .* v1)
        mul!(ψ, view(Ks.V, :, 1:mk), y, β, false)
        err += e
        remaining = abs(remaining - dt) ≤ 1e-14 * abs(δ) ? 0.0 : remaining - dt
    end
    return err, nmv
end

function propagate_block(H, Ψ0, ts, alg::ExpUtilsLanczosPropagatorAlg)
    Hop = _operator(H)
    tsv = _tvec(ts)
    δs = _increments(tsv)
    N, p = size(Ψ0)
    m = min(alg.m, N)
    L = sum(abs, δs; init=0.0)
    tol_rate = alg.tol / max(L, floatmin(Float64))
    Ks = ExponentialUtilities.KrylovSubspace{ComplexF64}(N, m)
    Us = [Matrix{ComplexF64}(undef, N, p) for _ in tsv]
    ψ = Vector{ComplexF64}(undef, N)
    nmv = 0
    errmax = 0.0
    for n in 1:p
        ψ .= view(Ψ0, :, n)
        errn = 0.0
        for (j, δ) in enumerate(δs)
            e, mv = _eu_lanczos_step!(ψ, Hop, δ, Ks, m, tol_rate, alg)
            errn += e
            nmv += mv
            Us[j][:, n] .= ψ
        end
        errmax = max(errmax, errn)
    end
    return Us, (; method=:expu_lanczos, err_est=errmax, nmatvec=nmv)
end

# -----------------------------------------------------------------------------
# 7b. ExponentialUtilities: expv_timestep (adaptive, native multi-time output)
# -----------------------------------------------------------------------------
"""
    ExpUtilsTimestepPropagatorAlg(; tol = 1e-10, m = 30, iop = 2, adaptive = true)
`expv_timestep(sorted |ts|, ∓iH, b; adaptive, tol, m, iop)` per column. ts must be real,
so the operator is skew-Hermitian; `iop = 2` (incomplete orthogonalization) is exact for
skew-Hermitian operators in exact arithmetic and costs like Lanczos. All times must share
a sign. No error estimate is returned.
"""
Base.@kwdef struct ExpUtilsTimestepPropagatorAlg <: AbstractStatePropagatorAlg
    tol::Float64 = 1e-10
    m::Int = 30
    iop::Int = 2
    adaptive::Bool = true
end

function propagate_block(H, Ψ0, ts, alg::ExpUtilsTimestepPropagatorAlg)
    Hop = _operator(H)
    tsv = _tvec(ts)
    N, p = size(Ψ0)
    s = all(≥(0), tsv) ? 1 : all(≤(0), tsv) ? -1 :
                             throw(ArgumentError("ExpUtilsTimestepPropagatorAlg: all times must have the same sign"))
    τ = abs.(tsv)
    nz = findall(!iszero, τ)
    perm = nz[sortperm(τ[nz])]
    Us = [Matrix{ComplexF64}(Ψ0) for _ in tsv]       # zero times stay Ψ0
    isempty(perm) && return Us, (; method=:expu_timestep, err_est=NaN, nmatvec=0)
    A = (-im * s) .* Hop
    τs = τ[perm]
    for n in 1:p
        U = ExponentialUtilities.expv_timestep(copy(τs), A, Vector{ComplexF64}(Ψ0[:, n]);
            adaptive=alg.adaptive, tol=alg.tol, m=min(alg.m, N),
            iop=alg.iop, ishermitian=false)
        U = reshape(U, N, :)
        for (jj, j) in enumerate(perm)
            Us[j][:, n] .= view(U, :, jj)
        end
    end
    return Us, (; method=:expu_timestep, err_est=NaN, nmatvec=-1)
end

# -----------------------------------------------------------------------------
# 8. Automatic selection (cost model; calibrate sec_* with the benchmark script)
# -----------------------------------------------------------------------------
"""
    AutoPropagatorAlg(; tol = 1e-10, small_dim = 200, max_diag_dim = 4000,
                        sec_eig = 4e-10, sec_gemm = 4e-11, sec_spmm = 1.5e-9,
                        sec_vec = 1e-9, real_eig_factor = 0.3, lanczos_steps = 30)
Chooses between diagonalization and block Chebyshev:
  cost_diag = f·sec_eig·N³ + sec_gemm·N²·p·(1 + n_t)      (Inf if N > max_diag_dim)
  cost_cheb = matvecs·(nnz·p·sec_spmm + 3Np·sec_vec) + axpys·Np·sec_vec
with matvecs/axpys taken from the actual Chebyshev plan (spectral bounds, tol, times).
"""
Base.@kwdef struct AutoPropagatorAlg <: AbstractStatePropagatorAlg
    tol::Float64 = 1e-10
    small_dim::Int = 200
    max_diag_dim::Int = 4000
    sec_eig::Float64 = 4e-10
    sec_gemm::Float64 = 1.6e-10
    sec_spmm::Float64 = 1.3e-9
    sec_vec::Float64 = 6.6e-10
    real_eig_factor::Float64 = 0.35
    lanczos_steps::Int = 30
end

function select_propagator(H, Ψ0, ts, alg::AutoPropagatorAlg)
    Hop = _operator(H)
    tsv = _tvec(ts)
    N, p = size(Ψ0)
    N ≤ alg.small_dim && return DiagonalizationPropagatorAlg(), (; cost_diag=NaN, cost_cheb=NaN)
    lo, hi = lanczos_bounds(Hop; steps=alg.lanczos_steps)
    cheb = ChebyshevPropagatorAlg(; tol=alg.tol, bounds=(lo, hi),
        lanczos_steps=alg.lanczos_steps)
    _, r, _ = chebyshev_interval(Hop, cheb)
    w = _nnz_per_row(Hop)
    plan = _chebyshev_plan(tsv, r, cheb, w)
    cost_cheb = plan.matvecs * (w * N * p * alg.sec_spmm + 3N * p * alg.sec_vec) +
        plan.axpys * N * p * alg.sec_vec
    f = eltype(H) <: Real ? alg.real_eig_factor : 1.0
    cost_diag = N > alg.max_diag_dim ? Inf :
                f * alg.sec_eig * float(N)^3 + alg.sec_gemm * float(N)^2 * p * (1 + length(tsv))
    chosen = cost_diag ≤ cost_cheb ? DiagonalizationPropagatorAlg() : cheb
    return chosen, (; cost_diag, cost_cheb)
end

function propagate_block(H, Ψ0, ts, alg::AutoPropagatorAlg)
    chosen, costs = select_propagator(H, Ψ0, ts, alg)
    Us, info = propagate_block(H, Ψ0, ts, chosen)
    return Us, merge(info, costs)
end

# -----------------------------------------------------------------------------
# 9. Legacy algorithms (kept for benchmarking; t = 0 crash fixed)
# -----------------------------------------------------------------------------
struct KrylovPropagatorAlg <: AbstractStatePropagatorAlg
    krylov_dim::Int
    tol::Float64            # NOTE: happy-breakdown threshold, not an accuracy target
end
KrylovPropagatorAlg(; krylov_dim=200, tol=1e-6) = KrylovPropagatorAlg(krylov_dim, tol)

function propagate_block(H, Ψ0, ts, alg::KrylovPropagatorAlg)
    tsv = _tvec(ts)
    N, p = size(Ψ0)
    iH = -im .* H
    Ks = KrylovSubspace{ComplexF64}(N, alg.krylov_dim)
    nmv = 0
    Us = map(tsv) do t
        iszero(t) && return Matrix{ComplexF64}(Ψ0)
        stack(1:p) do n
            arnoldi!(Ks, iH, Vector{ComplexF64}(Ψ0[:, n]); tol=alg.tol)
            nmv += Ks.m
            expv(t, Ks)
        end
    end
    return Us, (; method=:krylov_legacy, err_est=NaN, nmatvec=nmv)
end

# -----------------------------------------------------------------------------
# 10. System-level entry point (default is now AutoPropagatorAlg)
# -----------------------------------------------------------------------------
function scrambling_map(sys::QuantumDotSystem, measurements, ψres,
    hamiltonian, t, alg=AutoPropagatorAlg())
    scrambling_map(sys.H_main, sys.H_res, sys.H_total, measurements, ψres,
        hamiltonian, t, alg)
end

@testitem "Scrambling maps" begin
    using LinearAlgebra, SparseArrays, Random
    using QDELM
    const Q = QDELM
    using SpecialFunctions: besselj          # test-only dependency

    # ------------------------------- helpers -------------------------------------
    function random_hamiltonian(N; nnz_per_row=10, real_ham=false, seed=1)
        rng = Xoshiro(seed)
        T = real_ham ? Float64 : ComplexF64
        A = sprand(rng, T, N, N, nnz_per_row / (2N))
        nonzeros(A) .-= real_ham ? 0.5 : 0.5 + 0.5im
        return A + A' + spdiagm(0 => randn(rng, N))
    end
    random_isometry(N, p; seed=2) = Matrix(qr(randn(Xoshiro(seed), ComplexF64, N, p)).Q)[:, 1:p]

    function exact_propagation(H, Ψ0, ts)
        F = eigen(Hermitian(Matrix(H)))
        C = F.vectors' * Ψ0
        [F.vectors * (cis.(-t .* F.values) .* C) for t in ts]
    end
    maxcolerr(Us, Vs) = maximum(maximum(norm, eachcol(U - V)) for (U, V) in zip(Us, Vs))
    spectral_halfwidth(H) = (λ=eigvals(Hermitian(Matrix(H))); (λ[end] - λ[1]) / 2)

    const N = 500
    const P = 6
    const H = random_hamiltonian(N)
    const Hr = random_hamiltonian(N; real_ham=true, seed=3)
    const Ψ0 = random_isometry(N, P)

    time_sets(ρ) = [
        "zero" => [0.0],
        "single short" => [2.0 / ρ],
        "single long" => [300.0 / ρ],
        "negative" => [-15.0 / ρ],
        "uniform grid" => collect(range(5.0 / ρ, 50.0 / ρ; length=10)),
        "grid from 0" => collect(range(0.0, 40.0 / ρ; length=9)),
    ]

    const ALGS = [
        ("diag", Q.DiagonalizationPropagatorAlg(), 1e-10),
        ("cheb", Q.ChebyshevPropagatorAlg(tol=1e-11), 1e-9),
        ("cheb-gersh", Q.ChebyshevPropagatorAlg(tol=1e-11, bounds=:gershgorin), 1e-9),
        ("cheb-onepass", Q.ChebyshevPropagatorAlg(tol=1e-11, multitime=:onepass), 1e-9),
        ("cheb-stepping", Q.ChebyshevPropagatorAlg(tol=1e-11, multitime=:stepping), 1e-9),
        ("eu-lanczos", Q.ExpUtilsLanczosPropagatorAlg(tol=1e-11), 1e-8),
        ("eu-timestep", Q.ExpUtilsTimestepPropagatorAlg(tol=1e-11), 1e-6),
        ("auto", Q.AutoPropagatorAlg(tol=1e-11), 1e-9),
        ("auto-cheb", Q.AutoPropagatorAlg(tol=1e-11, small_dim=0, sec_eig=1.0), 1e-9),
    ]

    # ------------------------------- tests ---------------------------------------
    @testset "scrambling_map propagators" begin

        @testset "Bessel sequence (Miller) vs SpecialFunctions" begin
            for x in (1e-9, 0.37, 1.0, 7.5, 42.0, 250.0, 2500.0)
                J = Q.bessel_j_sequence(x)
                kmax = min(length(J) - 1, 600)
                @test maximum(abs(J[k+1] - besselj(k, x)) for k in 0:kmax) ≤ 1e-12
            end
        end

        @testset "Chebyshev coefficients (Jacobi–Anger identities)" begin
            for x in (0.0, 0.5, -3.0, 40.0, -900.0)
                a, tail = Q.chebyshev_coefficients(x, 1e-13)
                K = length(a) - 1
                @test tail ≤ 1e-13
                @test abs(sum(a) - cis(-x)) ≤ 1e-11                                   # y = 1
                @test abs(sum(a[k+1] * (-1)^k for k in 0:K) - cis(x)) ≤ 1e-11       # y = -1
                θ = 0.73
                @test abs(sum(a[k+1] * cos(k * θ) for k in 0:K) - cis(-x * cos(θ))) ≤ 1e-11
            end
        end

        @testset "spectral bounds enclose the spectrum" begin
            for HH in (H, Hr, Matrix(H))
                λ = eigvals(Hermitian(Matrix(HH)))
                g = Q.gershgorin_bounds(HH)
                @test g[1] ≤ λ[1] && λ[end] ≤ g[2]
                lo, hi = Q.lanczos_bounds(HH)
                @test lo ≤ λ[1] + 1e-8 && λ[end] - 1e-8 ≤ hi
            end
        end

        @testset "accuracy vs exact diagonalization" begin
            for (Hname, HH) in ("complex H" => H, "real H" => Hr)
                ρ = spectral_halfwidth(HH)
                for (tname, ts) in time_sets(ρ)
                    ref = exact_propagation(HH, Ψ0, ts)
                    for (name, alg, atol) in ALGS
                        name == "legacy-krylov" && tname == "single long" && continue
                        @testset "$Hname | $tname | $name" begin
                            Us, info = Q.propagate_block(HH, Ψ0, ts, alg)
                            @test length(Us) == length(ts)
                            @test all(size(U) == size(Ψ0) for U in Us)
                            @test maxcolerr(Us, ref) ≤ atol
                        end
                    end
                end
            end
        end

        @testset "Chebyshev error bound is a true upper bound" begin
            ρ = spectral_halfwidth(H)
            for tol in (1e-4, 1e-7, 1e-10), mode in (:onepass, :stepping),
                ts in ([37.0 / ρ], collect(range(3 / ρ, 60 / ρ; length=7)))

                alg = Q.ChebyshevPropagatorAlg(; tol, multitime=mode)
                Us, info = Q.propagate_block(H, Ψ0, ts, alg)
                err = maxcolerr(Us, exact_propagation(H, Ψ0, ts))
                @test err ≤ info.err_est + 1e-11
                @test info.err_est ≤ 1.5tol + 1e-11
            end
        end

        @testset "Chebyshev recovers from wrong spectral bounds" begin
            λ = eigvals(Hermitian(Matrix(H)))
            c, r = (λ[1] + λ[end]) / 2, (λ[end] - λ[1]) / 2
            ts = [25.0 / r]
            alg = Q.ChebyshevPropagatorAlg(tol=1e-11, bounds=(c - r / 3, c + r / 3))
            Us, info = Q.propagate_block(H, Ψ0, ts, alg)
            @test info.restarted
            @test info.guaranteed
            @test maxcolerr(Us, exact_propagation(H, Ψ0, ts)) ≤ 1e-9
        end

        @testset "isometry preserved" begin
            ts = [80.0 / spectral_halfwidth(H)]
            for alg in (Q.ChebyshevPropagatorAlg(), Q.ExpUtilsLanczosPropagatorAlg())
                Us, _ = Q.propagate_block(H, Ψ0, ts, alg)
                @test opnorm(Us[1]' * Us[1] - I) ≤ 1e-8
            end
        end

        @testset "AutoPropagatorAlg selection" begin
            ts = [50.0 / spectral_halfwidth(H)]
            _, i1 = Q.propagate_block(H, Ψ0, ts, Q.AutoPropagatorAlg(small_dim=size(H, 1)))
            @test i1.method == :diagonalization                                   # N ≤ small_dim
            _, i2 = Q.propagate_block(H, Ψ0, ts, Q.AutoPropagatorAlg(small_dim=0, sec_eig=1.0))
            @test i2.method == :chebyshev
            _, i3 = Q.propagate_block(H, Ψ0, ts, Q.AutoPropagatorAlg(small_dim=0, sec_eig=0.0, sec_gemm=0.0))
            @test i3.method == :diagonalization
            _, i4 = Q.propagate_block(H, Ψ0, ts, Q.AutoPropagatorAlg(small_dim=0, max_diag_dim=10))
            @test i4.method == :chebyshev
        end

        @testset "no global RNG consumption; thread safety" begin
            ts = [30.0 / spectral_halfwidth(H)]
            for alg in (Q.ChebyshevPropagatorAlg(), Q.AutoPropagatorAlg(small_dim=0, sec_eig=1.0))
                Random.seed!(42)
                a = rand()
                Random.seed!(42)
                Q.propagate_block(H, Ψ0, ts, alg)
                b = rand()
                @test a == b
            end
            tasks = [Threads.@spawn Q.propagate_block(H, Ψ0, ts, Q.ChebyshevPropagatorAlg())[1][1] for _ in 1:4]
            res = fetch.(tasks)
            @test all(==(res[1]), res)
        end

    end # testset
end
# ------------------- optional: check on a real QDELM system -------------------
"""
    check_scrambling_map(sys, measurements, ψres, H, ts; atol = 1e-7)
Compares scrambling_map for every path against diagonalization on a real system.
Example:
    grid = QDELM.generate_grid(2, nbr_dots_res)
    sys = QDELM.tight_binding_system(grid, qn_res)
    measurements = QDELM.charge_probabilities(sys)
    hams = QDELM.matrix_representation_hams(QDELM.hamiltonians(grid, ham_param_funcs), sys)
    ψres = QDELM.ground_state(hams.res)
    check_scrambling_map(sys, measurements, ψres, hams.total, [1.0, 2.0])
"""
function check_scrambling_map(sys, measurements, ψres, H, ts; atol=1e-7, include_block=true)
    ref = Q.scrambling_map(sys, measurements, ψres, H, ts, Q.DiagonalizationPropagatorAlg())
    @testset "scrambling_map on QDELM system" begin
        for (name, alg, _) in ALGS
            @testset "$name" begin
                S = Q.scrambling_map(sys, measurements, ψres, H, ts, alg)
                @test size(S) == size(ref)
                @test maximum(abs, S - ref) ≤ atol
            end
        end
        if include_block && size(H, 1) ≤ 2000       # independent operator-path cross-check
            S = Q.scrambling_map(sys, measurements, ψres, H, ts, Q.BlockPropagatorAlg())
            @test maximum(abs, S - ref) ≤ atol
        end
    end
end