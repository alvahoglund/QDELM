@testitem "Measurement layout" begin
    ms = ChargeMeasurements012(2, 2)
    gs = QDELM.groups(ms)
    @test length(gs) == 4
    @test vcat(gs...) |> sort == collect(1:QDELM.nrows(ms))
    @test gs[1] == [1, 3, 5]
    @test gs[3] == [7, 9, 11]
    R = QDELM.group_basis(ms)
    @test R' * R ≈ [1 0; 0 1]
    @test R' * ones(3) ≈ zeros(2) atol = 1e-12
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
        @test cov(E[g, :]') ≈ σ^2 * (I - ones(3, 3) / 3) atol = 2e-4
    end

    Σ = zeros(K, K)
    A = randn(3, 3)
    P = I - ones(3, 3) / 3
    Σ[1:3, 1:3] = 1e-2 * P * (A * A') * P # rank-deficient PSD block
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
end

@testitem "Naive-noise theory matches the iid formula" begin
    using LinearAlgebra
    K, d2 = 40, 16
    S = randn(K, d2)
    B = Matrix(qr(randn(d2, d2 - 1)).Q)[:, 1:(d2 - 1)]
    Σ = randn(d2, 3)
    b, σE = 0.02, 0.05
    U, D, V = svd(S * B)
    old = diag(Σ' * B * V * diagm((b * σE^2) ./ (b .* D .^ 2 .+ σE^2)) * V' * B' * Σ)
    @test mse_theory(S, B, Σ, b, NaiveNoise(σE)) ≈ old
    # Γ = b S_B S_B' + σE² I gives the same weights as the closed form
    A = U * diagm((b * D .^ 2) ./ (b .* D .^ 2 .+ σE^2)) * U'
    W_old = Σ' * B * pinv(S * B) * A
    @test W̃X_theory(S, B, Σ, b, NaiveNoise(σE)) ≈ W_old
end

@testitem "Isotropic noise on a complete POVM equals naive formula" begin
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
    Σ = randn(16, 2)
    b, σ = 0.02, 0.05
    @test mse_theory(S, B, Σ, b, IsotropicNoise(σ, ms)) ≈ mse_theory(S, B, Σ, b, NaiveNoise(σ))
end
