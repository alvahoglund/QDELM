using Test, LinearAlgebra, SparseArrays, Random
using QDELM, QuantumPropagators, ExponentialUtilities
const Q = QDELM

function random_hamiltonian(N; nnz_per_row = 10, real_ham = false, seed = 1)
    rng = Xoshiro(seed)
    T = real_ham ? Float64 : ComplexF64
    A = sprand(rng, T, N, N, nnz_per_row / (2N))
    nonzeros(A) .-= real_ham ? 0.5 : 0.5 + 0.5im
    return A + A' + spdiagm(0 => randn(rng, N))
end
random_isometry(N, p; seed = 2) = Matrix(qr(randn(Xoshiro(seed), ComplexF64, N, p)).Q)[:, 1:p]
function exact_propagation(H, Ψ0, ts)
    F = eigen(Hermitian(Matrix(H)))
    C = F.vectors' * Ψ0
    [F.vectors * (cis.(-t .* F.values) .* C) for t in ts]
end
maxcolerr(Us, Vs) = maximum(maximum(norm, eachcol(U - V)) for (U, V) in zip(Us, Vs))
spectral_halfwidth(H) = (λ = eigvals(Hermitian(Matrix(H))); (λ[end] - λ[1]) / 2)

const N = 300
const P = 4
const H  = random_hamiltonian(N)
const Hr = random_hamiltonian(N; real_ham = true, seed = 3)
const Ψ0 = random_isometry(N, P)

time_sets(ρ) = [
    "zero"         => [0.0],
    "single short" => [2.0 / ρ],
    "single long"  => [300.0 / ρ],
    "negative"     => [-15.0 / ρ],
    "uniform grid" => collect(range(5.0 / ρ, 50.0 / ρ; length = 10)),
    "grid from 0"  => collect(range(0.0, 40.0 / ρ; length = 9)),
    "non-uniform"  => [1.0, 4.0, 4.5, 20.0] ./ ρ,
]

const QP_ALGS = [
    ("qp-cheby",         Q.QPAlg(Q.QPCheby(cheby_coeffs_limit = 1e-13)),                        1e-9),
    ("qp-cheby-gersh",   Q.QPAlg(Q.QPCheby(cheby_coeffs_limit = 1e-13, bounds = :gershgorin)),  1e-9),
    ("qp-cheby-qpspec",  Q.QPAlg(Q.QPCheby(cheby_coeffs_limit = 1e-13, bounds = :qp)),          1e-8),
    ("qp-cheby-sub",     Q.QPAlg(Q.QPCheby(cheby_coeffs_limit = 1e-13, max_ρdt = 20.0)),       1e-9),
    ("qp-cheby-block",   Q.QPAlg(Q.QPCheby(cheby_coeffs_limit = 1e-13); block = true),         1e-9),
    ("qp-cheby-noreuse", Q.QPAlg(Q.QPCheby(cheby_coeffs_limit = 1e-13); reuse = false),        1e-9),
    ("qp-newton",        Q.QPAlg(Q.QPNewton(relerr = 1e-12)),                                    1e-8),
    ("qp-newton-block",  Q.QPAlg(Q.QPNewton(relerr = 1e-12); block = true),                     1e-8),
    ("qp-expu",          Q.QPAlg(Q.QPExpUtils(tol = 1e-12)),                                     1e-8),
    ("qp-expprop",       Q.QPAlg(Q.QPExpProp()),                                                 1e-9),
    ("qp-expprop-block", Q.QPAlg(Q.QPExpProp(); block = true),                                   1e-9),
]

@testset "QuantumPropagators path" begin

@testset "time-grid runs" begin
    r = Q._qp_runs([1.0, 2.0, 3.0], 10.0, Inf)
    @test length(r) == 1 && r[1].count == 3 && r[1].nsub == 1 && length(r[1].tlist) == 4
    r = Q._qp_runs([0.0, 0.5, 1.0], 10.0, 2.0)
    @test r[1].nsub == 0 && r[1].count == 1
    @test r[2].count == 2 && r[2].nsub == 3               # ceil(10·0.5/2)
    r = Q._qp_runs([2.0], 10.0, 4.0)
    @test r[1].nsub == 5 && all(d -> isapprox(d, 0.4), diff(r[1].tlist))
    r = Q._qp_runs([1.0, 1.5, 3.0], 10.0, Inf)
    @test length(r) == 3
    for ts in ([0.3], collect(0.0:0.25:2.0), [0.1, 0.2, 0.7, 0.7, 3.0])
        for run in Q._qp_runs(ts, 7.0, 1.5)
            run.nsub == 0 && continue
            @test 7.0 * run.δ / run.nsub ≤ 1.5 + 1e-12
            @test run.tlist[end] ≈ run.count * run.δ
        end
    end
end

@testset "accuracy vs exact diagonalization" begin
    for (Hname, HH) in ("complex H" => H, "real H" => Hr)
        ρ = spectral_halfwidth(HH)
        for (tname, ts) in time_sets(ρ)
            ref = exact_propagation(HH, Ψ0, ts)
            for (name, alg, atol) in QP_ALGS
                @testset "$Hname | $tname | $name" begin
                    Us, info = Q.propagate_block(HH, Ψ0, ts, alg)
                    @test length(Us) == length(ts)
                    @test maxcolerr(Us, ref) ≤ atol
                end
            end
        end
    end
end

@testset "block == column-wise; agrees with native Chebyshev" begin
    ts = collect(range(3.0, 30.0; length = 4)) ./ spectral_halfwidth(H)
    Ucol, _ = Q.propagate_block(H, Ψ0, ts, Q.QPAlg(Q.QPCheby(cheby_coeffs_limit = 1e-14)))
    Ublk, _ = Q.propagate_block(H, Ψ0, ts, Q.QPAlg(Q.QPCheby(cheby_coeffs_limit = 1e-14); block = true))
    Unat, _ = Q.propagate_block(H, Ψ0, ts, Q.ChebyshevPropagatorAlg(tol = 1e-13))
    @test maxcolerr(Ucol, Ublk) ≤ 1e-10
    @test maxcolerr(Ucol, Unat) ≤ 1e-9
end

@testset "substep bookkeeping" begin
    t = 100.0 / spectral_halfwidth(H)
    _, info = Q.propagate_block(H, Ψ0, [t], Q.QPAlg(Q.QPCheby(max_ρdt = 20.0)))
    @test info.nsteps == P * ceil(Int, info.ρ * t / 20.0)
end

@testset "argument errors" begin
    @test_throws ArgumentError Q.propagate_block(H, Ψ0, [1.0], Q.QPAlg(Q.QPExpUtils(); block = true))
    @test_throws ArgumentError Q.propagate_block(H, Ψ0, [-1.0, 1.0], Q.QPAlg())
end

@testset "default QPCheby does not consume the global RNG" begin
    ts = [20.0 / spectral_halfwidth(H)]
    Random.seed!(7); a = rand()
    Random.seed!(7); Q.propagate_block(H, Ψ0, ts, Q.QPAlg()); b = rand()
    @test a == b
end

@testset "QP Cheby has no bound recovery (documents the difference)" begin
    λ = eigvals(Hermitian(Matrix(H)))
    c, r = (λ[1] + λ[end]) / 2, (λ[end] - λ[1]) / 2
    ts = [25.0 / r]
    bad = Q.QPAlg(Q.QPCheby(bounds = (c - r / 3, c + r / 3)))
    failed = try
        Us, _ = Q.propagate_block(H, Ψ0, ts, bad)
        maxcolerr(Us, exact_propagation(H, Ψ0, ts)) > 1e-6
    catch
        true
    end
    @test failed      # ChebyshevPropagatorAlg restarts with Gershgorin bounds instead
end

end # testset