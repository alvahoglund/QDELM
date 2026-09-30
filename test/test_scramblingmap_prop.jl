
@testitem "Scrambling maps" begin
    using LinearAlgebra, SparseArrays, Random
    using QDELM
    Q = QDELM
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

    N = 500
    P = 6
    H = random_hamiltonian(N)
    Hr = random_hamiltonian(N; real_ham=true, seed=3)
    Ψ0 = random_isometry(N, P)

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