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
        ("auto-cheb", Q.AutoPropagatorAlg(tol=1e-11, candidates=(:cheby,)), 1e-9),
        ("auto-diag", Q.AutoPropagatorAlg(candidates=(:diag,)), 1e-12),
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
            # Helper to inspect selection report
            sel(Ψ, ts; kw...) = Q.select_propagator(H, Ψ, ts, Q.AutoPropagatorAlg(; kw...))[2]

            # Approximate half-width for scale-setting
            evals = eigvals(Hermitian(Matrix(H)))
            ρ = (maximum(evals) - minimum(evals)) / 2
            long_times = [150.0 / ρ, 300.0 / ρ]

            # 1. Trivial time vectors (all zeros) bypass propagation entirely
            @test sel(Ψ0, [0.0, 0.0]).reason == :no_propagation

            # 2. Dimensions under small_dim pick diagonalization immediately
            @test sel(Ψ0, long_times; small_dim=size(H, 1)).choice == :diag
            @test sel(Ψ0, long_times; small_dim=size(H, 1)).reason == :small

            # 3. If diagonalization is cheaper than specrange estimation itself, choose :diag
            #    without calling specrange
            rep_cheap = sel(Ψ0, long_times; small_dim=0, sec_eig=0.0, sec_gemm=0.0)
            @test rep_cheap.choice == :diag
            @test rep_cheap.reason == :cheaper_than_specrange

            # 4. Forcing candidates = (:cheby,) or setting max_diag_dim = 0 picks Chebychev
            rep_cheby = sel(Ψ0, long_times; max_diag_dim=0)
            @test rep_cheby.choice == :cheby
            @test rep_cheby.reason == :cost
            @test isfinite(rep_cheby.cost_cheby)
            @test isinf(rep_cheby.cost_diag)

            # 5. Chebychev cost grows monotonically with ρ·T
            s_short = sel(Ψ0, [10.0 / ρ]; max_diag_dim=0)
            s_long = sel(Ψ0, [100.0 / ρ]; max_diag_dim=0)
            @test s_long.cost_cheby > s_short.cost_cheby
            @test s_long.ρT > s_short.ρT

            # 6. Reusing spectral range: chosen QPAlg must inherit manual bounds
            alg_chosen, rep = Q.select_propagator(H, Ψ0, long_times, Q.AutoPropagatorAlg(max_diag_dim=0))
            @test alg_chosen isa Q.QPAlg
            @test alg_chosen.kwargs.specrange_method == :manual
            @test alg_chosen.kwargs.E_min < alg_chosen.kwargs.E_max

            # 7. Error thrown if no valid candidate is available
            @test_throws ArgumentError Q.select_propagator(
                H, Ψ0, long_times,
                Q.AutoPropagatorAlg(max_diag_dim=0, candidates=(:diag,))
            )

            # 8. End-to-end propagation accuracy (mixed forward/backward times)
            test_ts = [-20.0 / ρ, 0.0, 35.0 / ρ]
            Us, info = Q.propagate_block(
                H, Ψ0, test_ts,
                Q.AutoPropagatorAlg(tol=1e-11, max_diag_dim=0)
            )
            @test maxcolerr(Us, exact_propagation(H, Ψ0, test_ts)) ≤ 1e-8
            @test haskey(info, :selection)
            @test info.selection.choice === :cheby
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