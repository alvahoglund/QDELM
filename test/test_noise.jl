@testitem "Measurement layout" begin
    ms = ChargeMeasurements012(2, 2)
    gs = QDELM.groups(ms)
    @test length(gs) == 4
    @test vcat(gs...) |> sort == collect(1:QDELM.nrows(ms))
    @test gs[1] == [1, 3, 5]
    @test gs[3] == [7, 9, 11]
end

@testitem "Noise sampling" begin
    using LinearAlgebra, Statistics, Random
    Random.seed!(1)
    ms = ChargeMeasurements012(2, 1)
    K = QDELM.nrows(ms)
    X = zeros(K, 10^5)

    E = noise_sample(NaiveNoise(0.1), X)
    @test std(E) ≈ 0.1 rtol = 0.02

    σ = 0.1
    E = noise_sample(IsotropicNoise(σ, ms), X)
    for g in QDELM.groups(ms)
        @test maximum(abs, sum(E[g, :], dims = 1)) < 1e-12
        @test cov(E[g, :]') ≈ σ^2 * (1.5I - fill(0.5, 3, 3)) atol = 2e-4
    end

    ms01 = ChargeMeasurements01(2, 1)
    E = noise_sample(IsotropicNoise(σ, ms01), zeros(QDELM.nrows(ms01), 10^5))
    for g in QDELM.groups(ms01)
        @test cov(E[g, :]') ≈ σ^2 * [1 -0.5; -0.5 1] atol = 2e-4
    end

    Σ = zeros(K, K)
    A = randn(3, 3)
    Q = I - ones(3, 3) / 3
    Σ[1:3, 1:3] = 1e-2 * Q * (A * A') * Q # rank-deficient PSD block
    Σ[4:6, 4:6] = 1e-2 * Matrix(I, 3, 3)
    E = noise_sample(CovariantNoise(Σ), X)
    @test cov(E') ≈ Σ atol = 1e-3
end

@testitem "Shot noise sampling" begin
    using LinearAlgebra, Statistics, Random
    Random.seed!(2)
    ms = ChargeMeasurements012(1, 1)
    p = [0.2, 0.5, 0.3]
    n_s = 100
    X = repeat(p, 1, 10^5)
    E = noise_sample(ShotNoise(n_s, ms), X)
    @test maximum(abs, sum(E, dims = 1)) < 1e-12
    @test cov(E') ≈ (Diagonal(p) - p * p') / n_s atol = 2e-4

    E = noise_sample(ShotNoise(n_s, ChargeMeasurements01(1, 1)), X[1:2, :])
    @test cov(E') ≈ (Diagonal(p[1:2]) - p[1:2] * p[1:2]') / n_s atol = 2e-4
end

@testitem "Naive-noise theory matches the iid formula" begin
    using LinearAlgebra
    K, d2 = 40, 16
    S = randn(K, d2)
    B = Matrix(qr(randn(d2, d2 - 1)).Q)[:, 1:(d2 - 1)]
    P = randn(d2, 3)
    b, σE = 0.02, 0.05
    U, D, V = svd(S * B)
    old = diag(P' * B * V * diagm((b * σE^2) ./ (b .* D .^ 2 .+ σE^2)) * V' * B' * P)
    @test mse_theory(S, B, P, b, NaiveNoise(σE)) ≈ old
    # Γ = b S_B S_B' + σE² I gives the same weights as the closed form
    A = U * diagm((b * D .^ 2) ./ (b .* D .^ 2 .+ σE^2)) * U'
    W_old = P' * B * pinv(S * B) * A
    @test W̃X_theory(S, B, P, b, NaiveNoise(σE)) ≈ W_old
end

@testitem "Isotropic noise on a complete POVM equals rescaled naive formula" begin
    using LinearAlgebra
    M = 8
    ms = ChargeMeasurements012(M, 1)
    K = 3M
    # S_B rows lie in the sum-zero plane of each group
    S = randn(K, 16)
    for g in QDELM.groups(ms)
        S[g, :] .-= sum(S[g, :], dims = 1) / 3
    end
    B = Matrix(1.0I, 16, 15)
    P = randn(16, 2)
    b, σ = 0.02, 0.05
    @test mse_theory(S, B, P, b, IsotropicNoise(σ, ms)) ≈
          mse_theory(S, B, P, b, NaiveNoise(σ * sqrt(3 / 2)))
end

@testitem "Dropping outcome 2 loses no information" begin
    using LinearAlgebra
    sys = tight_binding_system(2, 2, 1)
    pf = QDELM.random_param_functions(; u_intra = 1.0, t = 1.0, t_so = 1.0, u_inter = 1.0)
    hams = QDELM.matrix_representation_hams(hamiltonians(sys.grids, pf), sys)
    ψ_res = QDELM.eig_state(hams.res, 2)
    ts = [5.0, 10.0]
    alg = QDELM.DiagonalizationPropagatorAlg()
    ms01, ms012 = ChargeMeasurements01(sys, ts), ChargeMeasurements012(sys, ts)
    S01 = scrambling_map(sys, ms01, ψ_res, hams.total, ts, alg)
    S012 = scrambling_map(sys, ms012, ψ_res, hams.total, ts, alg)
    Pm, _ = QDELM.pauli_matrix(sys.Hs_main, sys.H_main)
    B = Pm[:, 2:end] / 2
    b = 0.0147
    F(n, S) = QDELM.information_matrix(n, S, B, b)

    @test F(IsotropicNoise(1, ms01), S01) ≈ F(IsotropicNoise(1, ms012), S012)
    @test F(IsotropicNoise(1, ms012), S012) ≈ 2 / 3 * F(NaiveNoise(1), S012)
    @test F(ShotNoise(1, ms01), S01) ≈ F(ShotNoise(1, ms012), S012)
end

@testitem "Measurement set layout matches scrambling_map" begin
    sys = tight_binding_system(2, 2, 1)
    pf = QDELM.random_param_functions(; u_intra = 1.0, t = 1.0, t_so = 1.0, u_inter = 1.0)
    hams = QDELM.matrix_representation_hams(hamiltonians(sys.grids, pf), sys)
    ψ_res = QDELM.eig_state(hams.res, 2)
    ts = [5.0, 10.0]
    alg = QDELM.DiagonalizationPropagatorAlg()

    ms = ChargeMeasurements012(sys, ts)
    @test ms.M == length(sys.grids.total) && ms.n_times == 2
    S = scrambling_map(sys, ms, ψ_res, hams.total, ts, alg)
    @test size(S, 1) == QDELM.nrows(ms)
    @test S ≈
          scrambling_map(sys, QDELM.charge_probabilities(sys), ψ_res, hams.total, ts, alg)

    @test_throws ArgumentError scrambling_map(
        sys, ChargeMeasurements012(sys, [5.0]), ψ_res, hams.total, ts, alg)
    @test_throws ArgumentError QDELM.validate_layout(ChargeMeasurements012(ms.M, 1), S[1:(3 * ms.M), :] .*
                                                                                     2)

    ms01 = ChargeMeasurements01(sys, ts)
    S01 = scrambling_map(sys, ms01, ψ_res, hams.total, ts, alg)
    @test size(S01, 1) == QDELM.nrows(ms01)
    @test_throws DimensionMismatch QDELM.validate_layout(ms, S01)
end