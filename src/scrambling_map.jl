abstract type AbstractPropagatorAlg end
struct BlockPropagatorAlg <: AbstractPropagatorAlg end

function scrambling_map(H_main, H_res, H_total, measurements, ψres,
        hamiltonian, t::Number, ::BlockPropagatorAlg)
    ρ_res = density_matrix(ψres)
    U = propagator(t, hamiltonian)
    measurements_t = map(Base.Fix1(operator_time_evolution, U), measurements)
    eff_measurements = map(
        mt -> effective_measurement(mt, ρ_res, H_main, H_res, H_total),
        measurements_t)
    return reduce(vcat, (vec(m)' for m in eff_measurements))
end

"""
    KrylovPropagatorAlg(; krylov_dim = 200, tol = 1e-6)

Constructs propagator U = exp(-iHt)|ψ> using Krylov subspace methods from ExponentialUtilities. 
`krylov_dim` is the dimension of the Krylov subspace, and `tol` is the 
tolerance for convergence in the Arnoldi iteration.
Algorithm assumes pure reservoir state. 
"""
struct KrylovPropagatorAlg <: AbstractPropagatorAlg
    krylov_dim::Int
    tol::Float64
end
function KrylovPropagatorAlg(; krylov_dim = 200, tol = 1e-6)
    KrylovPropagatorAlg(krylov_dim, tol)
end
function scrambling_map(H_main, H_res, H_total, measurements, ψres::AbstractVector,
        hamiltonian, t::Number, alg::KrylovPropagatorAlg)
    iH = -im .* hamiltonian
    N = dim(H_total)
    N_main = dim(H_main)
    Ks = KrylovSubspace{ComplexF64}(N, alg.krylov_dim)
    e_j = zeros(ComplexF64, N_main)
    U = stack(1:N_main) do n
        fill!(e_j, 0)
        e_j[n] = 1.0
        ψtot = generalized_kron((e_j, ψres), (H_main, H_res) => H_total)
        arnoldi!(Ks, iH, ψtot; tol = alg.tol)
        expv(t, Ks)
    end
    stack(op -> vec(U' * Diagonal(op) * U), measurements)'
end

"""
    SteppingKrylovPropagatorAlg(; krylov_dim = 100, tol = 1e-6, step_size = NaN, step_size_const = 0.5)
Constructs propagator U = exp(-iHt)|ψ> using a stepping method together with Krylov subspace methods from ExponentialUtilities. 
`krylov_dim` is the dimension of the Krylov subspace, and `tol` is the tolerance for convergence in the Arnoldi iteration. 
`step_size` is the time step size for the stepping method. 
If `step_size` is NaN, it will be determined adaptively based on the spectral radius of the Hamiltonian and the Krylov dimension, using `step_size_const` as a scaling factor.
Algorithm assumes pure reservoir state. 
"""
struct SteppingKrylovPropagatorAlg <: AbstractPropagatorAlg
    krylov_dim::Int
    tol::Float64
    step_size::Float64
    step_size_const::Float64
    λ_center::Float64
end

function SteppingKrylovPropagatorAlg(;
        krylov_dim = 100, tol = 1e-6, step_size = NaN, step_size_const = 0.5)
    return SteppingKrylovPropagatorAlg(krylov_dim, tol, step_size, step_size_const, NaN)
end

function get_step_size(ham, alg::SteppingKrylovPropagatorAlg)
    if !isnan(alg.step_size) && !isnan(alg.λ_center)
        return alg.step_size, alg.λ_center
    end
    λ_large, _ = eigsolve(ham, rand(ComplexF64, size(ham, 1)), 1, :LR)
    λ_small, _ = eigsolve(ham, rand(ComplexF64, size(ham, 1)), 1, :SR)
    λ_min = real(λ_small[1])
    λ_max = real(λ_large[1])
    λ_center = (λ_max + λ_min) / 2
    if !isnan(alg.step_size)
        return alg.step_size, λ_center
    end
    spectral_radius = (λ_max - λ_min) / 2
    step_size = alg.step_size_const * alg.krylov_dim / spectral_radius
    return step_size, λ_center
end

function krylov_step(Ks, iH, ψ::AbstractVector, dt, tol)
    arnoldi!(Ks, iH, ψ; tol = tol)
    return expv(dt, Ks)
end

function expv_stepped(
        iH, ψtot::AbstractVector, t::Number,
        alg::SteppingKrylovPropagatorAlg, step_size, Ks)
    nbr_steps = ceil(Int, abs(t) / min(step_size, abs(t)))
    dt = t / nbr_steps
    ψ_step = ψtot
    for _ in 1:nbr_steps
        ψ_step = krylov_step(Ks, iH, ψ_step, dt, alg.tol)
        ψ_step ./= norm(ψ_step)
    end
    return ψ_step
end

function scrambling_map(H_main, H_res, H_total, measurements, ψres::AbstractVector,
        hamiltonian, t::Number, alg::SteppingKrylovPropagatorAlg)
    step_size, center = get_step_size(hamiltonian, alg)
    N = dim(H_total)
    N_main = dim(H_main)
    # Shift spectrum to center around zero: exp(-iHt) = exp(-i·center·t) · exp(-i(H-center·I)t)
    iH = -im .* hamiltonian + im * center * I
    Ks = KrylovSubspace{ComplexF64}(N, alg.krylov_dim)
    e_j = zeros(ComplexF64, N_main)
    U = stack(1:N_main) do n
        fill!(e_j, 0)
        e_j[n] = 1.0
        ψtot = generalized_kron((e_j, ψres), (H_main, H_res) => H_total)
        expv_stepped(iH, ψtot, t, alg, step_size, Ks)
    end
    U .*= exp(-im * center * t)
    stack(op -> vec(U' * Diagonal(op) * U), measurements)'
end
"""
    DiagonalizationAlg()
Constructs propagator U = exp(-iHt)|ψ> by diagonalizing the Hamiltonian.
Algorithm assumes pure reservoir state. 
"""
struct DiagonalizationPropagatorAlg <: AbstractPropagatorAlg end
function scrambling_map(H_main, H_res, H_total, measurements, ψres::AbstractVector,
        hamiltonian, t::Number, ::DiagonalizationPropagatorAlg)
    N_main = dim(H_main)
    F = eigen(Hermitian(Matrix(hamiltonian)))
    phases = exp.(-im .* F.values .* t)
    e_j = zeros(ComplexF64, N_main)
    U = stack(1:N_main) do n
        fill!(e_j, 0)
        e_j[n] = 1.0
        ψtot = generalized_kron((e_j, ψres), (H_main, H_res) => H_total)
        F.vectors * (phases .* (F.vectors' * ψtot))
    end
    stack(op -> vec(U' * Diagonal(op) * U), measurements)'
end

"""
    AdaptivePropagatorAlg(; krylov_dim = 100, tol = 1e-6, step_size_const = 0.5)
Algorithm that adaptively selects between Krylov, stepping Krylov, and diagonalization methods for computing the propagator U = exp(-iHt)|ψ>.
For short times, small spectral radius, Krylov methods are preferred. 
For longer times or larger spectral radius, exact diagonalization is used.
"""
struct AdaptivePropagatorAlg <: AbstractPropagatorAlg
    krylov_dim::Int
    tol::Float64
    step_size_const::Float64
end
function AdaptivePropagatorAlg(; krylov_dim = 100, tol = 1e-6, step_size_const = 0.5)
    AdaptivePropagatorAlg(krylov_dim, tol, step_size_const)
end
const DIAG_MAX_DIM = 3000
function scrambling_map(H_main, H_res, H_total, measurements, ψres::AbstractVector,
        hamiltonian, t::Number, alg::AdaptivePropagatorAlg)
    #Parameters for the stepping algorithm
    stepping_alg = SteppingKrylovPropagatorAlg(;
        krylov_dim = alg.krylov_dim, tol = alg.tol, step_size_const = alg.step_size_const)
    step_size, center = get_step_size(hamiltonian, stepping_alg)
    n_steps = ceil(Int, abs(t) / min(step_size, abs(t)))
    N = dim(H_total)

    alg = if n_steps <= 1
        KrylovPropagatorAlg(alg.krylov_dim, alg.tol)
    elseif N <= DIAG_MAX_DIM && n_steps > N ÷ alg.krylov_dim
        DiagonalizationPropagatorAlg()
    else
        SteppingKrylovPropagatorAlg(
            alg.krylov_dim, alg.tol, step_size, alg.step_size_const, center)
    end
    scrambling_map(H_main, H_res, H_total, measurements, ψres, hamiltonian, t, alg)
end

function scrambling_map(H_main, H_res, H_total, measurements,
        ψres, hamiltonian, t::AbstractArray, alg)
    mapreduce(
        ti -> scrambling_map(
            H_main, H_res, H_total, measurements, ψres, hamiltonian, ti, alg),
        vcat,
        t)
end

function scrambling_map(sys::QuantumDotSystem, measurements, ψres,
        hamiltonian, t, alg = AdaptivePropagatorAlg())
    scrambling_map(
        sys.H_main, sys.H_res, sys.H_total, measurements, ψres, hamiltonian, t, alg)
end