# =============================================================================
# noise.jl — measurement noise models: sampling and the matching MSE theory.
# Theory follows noise-covariance.md. Requires `using Distributions: Normal, MvNormal`.
# =============================================================================

## ================= Measurement layout =====================
"""
Row layout of S: index = (t-1)*n_outcomes*M + n*M + j with outcome n = 0:n_outcomes-1,
dot j = 1:M and time t = 1:n_times. Rows (t, ·, j) form one POVM ("group").
"""
abstract type MeasurementSet end

"Complete charge measurement (0, 1, 2 electrons): the three rows of a group sum to the identity."
struct ChargeMeasurements012 <: MeasurementSet
    M::Int
    n_times::Int
end

"Charge measurement with outcomes 0, 1 only: not a complete POVM, no sum-zero constraint."
struct ChargeMeasurements01 <: MeasurementSet
    M::Int
    n_times::Int
end

n_outcomes(::ChargeMeasurements012) = 3
n_outcomes(::ChargeMeasurements01) = 2
nrows(ms::MeasurementSet) = n_outcomes(ms) * ms.M * ms.n_times

function groups(ms::MeasurementSet)
    no = n_outcomes(ms)
    return [[(t - 1) * no * ms.M + n * ms.M + j for n in 0:(no - 1)]
            for t in 1:(ms.n_times) for j in 1:(ms.M)]
end

"Orthonormal basis (columns) of the space a group's noise lives in."
group_basis(::ChargeMeasurements012) = [1/√2 1/√6; -1/√2 1/√6; 0.0 -2/√6]
group_basis(::ChargeMeasurements01) = Matrix(1.0I, 2, 2)

## ================= Noise models =====================
abstract type NoiseModel end

"No noise."
struct NoNoise <: NoiseModel end

"Independent `Normal(0, σ)` noise on every entry."
struct NaiveNoise{T <: Real} <: NoiseModel
    σ::T
end

"Isotropic noise inside each group's constraint plane: covariance σ² Q (Q projects onto the group basis)."
struct IsotropicNoise{T <: Real, M <: MeasurementSet} <: NoiseModel
    σ::T
    ms::M
end

"Zero-mean Gaussian noise with a fixed (possibly singular) K×K covariance Σ."
struct CovariantNoise{T <: Real} <: NoiseModel
    Σ::Matrix{T}
    L::Matrix{T}   # Σ = L L'
end
function CovariantNoise(Σ::AbstractMatrix{<:Real})
    F = eigen(Symmetric(Matrix(Σ)))
    keep = F.values .> sqrt(eps()) * max(maximum(F.values), eps())
    all(F.values .> -sqrt(eps()) * max(maximum(F.values), eps())) ||
        throw(ArgumentError("Σ must be positive semidefinite"))
    L = F.vectors[:, keep] .* sqrt.(F.values[keep])'
    return CovariantNoise(Matrix(Σ), L)
end

"""
Multinomial shot noise, Gaussian approximation: per group Cov(e | p) = (diag p - p p')/n_s.
`n_s` is a number or a vector with one entry per group (see `groups`).
"""
struct ShotNoise{S <: Union{Real, AbstractVector{<:Real}}, M <: MeasurementSet} <: NoiseModel
    n_s::S
    ms::M
end

shots(n::ShotNoise{<:Real}, ::Int) = n.n_s
shots(n::ShotNoise{<:AbstractVector}, g::Int) = n.n_s[g]

## ================= Sampling =====================
"Noise matrix E with the size of X, columns are independent draws."
noise_sample(::NoNoise, X) = zeros(float(real(eltype(X))), size(X))
noise_sample(n::NaiveNoise, X) = rand(Normal(0, n.σ), size(X))

function noise_sample(n::IsotropicNoise, X)
    size(X, 1) == nrows(n.ms) || throw(DimensionMismatch("X has $(size(X, 1)) rows, expected $(nrows(n.ms))"))
    R = group_basis(n.ms)
    r = size(R, 2)
    dist = MvNormal(zeros(r), fill(float(n.σ), r))
    E = zeros(float(real(eltype(X))), size(X))
    for g in groups(n.ms)
        E[g, :] = R * rand(dist, size(X, 2))
    end
    return E
end

function noise_sample(n::CovariantNoise, X)
    size(X, 1) == size(n.Σ, 1) || throw(DimensionMismatch("X has $(size(X, 1)) rows, expected $(size(n.Σ, 1))"))
    r = size(n.L, 2)
    return n.L * rand(MvNormal(zeros(r), ones(r)), size(X, 2))
end

function noise_sample(n::ShotNoise, X)
    size(X, 1) == nrows(n.ms) || throw(DimensionMismatch("X has $(size(X, 1)) rows, expected $(nrows(n.ms))"))
    R = group_basis(n.ms)
    E = zeros(float(real(eltype(X))), size(X))
    μ = zeros(size(R, 2))
    for (gi, g) in enumerate(groups(n.ms)), c in axes(X, 2)
        p = max.(real.(@view X[g, c]), 1e-12)   # floor keeps the covariance positive definite
        Σ = (Diagonal(p) - p * p') / shots(n, gi)
        E[g, c] = R * rand(MvNormal(μ, Symmetric(R' * Σ * R)))
    end
    return E
end

add_noise(X, noise::NoiseModel) = X + noise_sample(noise, X)

add_noise!(X, ::NoNoise) = X
function add_noise!(X, noise::NoiseModel)
    X .+= noise_sample(noise, X)
    return X
end

## ================= Theory =====================
"Row-space covariance Σ_E (K×K) of the noise, averaged over the Hilbert–Schmidt ensemble (parameter `b`)."
noise_covariance(n::NaiveNoise, S, B, b) = n.σ^2 * I(size(S, 1))
noise_covariance(n::CovariantNoise, S, B, b) = n.Σ

function noise_covariance(n::IsotropicNoise, S, B, b)
    R = group_basis(n.ms)
    Σ = zeros(size(S, 1), size(S, 1))
    for g in groups(n.ms)
        Σ[g, g] = n.σ^2 * (R * R')
    end
    return Σ
end

function noise_covariance(n::ShotNoise, S, B, b)
    SB = to_real.(S * B)
    d = isqrt(size(S, 2))
    p̄ = to_real.(S * vec(Matrix(I, d, d))) ./ d
    Σ = zeros(size(S, 1), size(S, 1))
    for (gi, g) in enumerate(groups(n.ms))
        p = p̄[g]
        Σ[g, g] = (Diagonal(p) - p * p' - b * SB[g, :] * SB[g, :]') / shots(n, gi)
    end
    return Σ
end

"F = S_B† Σ_E⁺ S_B, the noise-whitened information matrix."
information_matrix(n::NaiveNoise, S, B, b) = (SB = S * B; SB' * SB / n.σ^2)
function information_matrix(n::NoiseModel, S, B, b)
    SB = S * B
    return SB' * pinv(Hermitian(noise_covariance(n, S, B, b))) * SB
end

"Per-target MSE of the optimal linear estimator, diag(Σ_B† (1/b + F)⁻¹ Σ_B)."
function mse_theory(S, B, Σ, b, noise::NoiseModel)
    F = information_matrix(noise, S, B, b)
    T = Σ' * B
    return to_real.(diag(T * inv(Hermitian(I / b + F)) * T'))
end

"Noisy weights W̃_X = b Σ_B† S_B† Γ⁺ with Γ = b S_B S_B† + Σ_E."
function W̃X_theory(S, B, Σ, b, noise::NoiseModel)
    SB = S * B
    Γ = b * SB * SB' + noise_covariance(noise, S, B, b)
    return b * (Σ' * B) * SB' * pinv(Hermitian(Γ))
end

mse_theory(S, B, Σ, b, ::NoNoise) = zeros(size(Σ, 2))
W̃X_theory(S, B, Σ, b, ::NoNoise) = (Σ' * B) * pinv(S * B)

"Whitened singular values sqrt(λ_p) of F and overlaps of the targets with its eigenvectors. For NaiveNoise(σ) these are σ_p/σ."
function sv_overlap(S, B, Σ, b, noise::NoiseModel)
    F = eigen(Hermitian(information_matrix(noise, S, B, b)))
    return (vals = sqrt.(max.(F.values, 0)), overlaps = Σ' * B * F.vectors)
end
