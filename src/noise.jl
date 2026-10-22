# =============================================================================
# noise.jl — measurement noise models: sampling and the matching MSE theory.
# Theory follows noise-covariance.md. Requires `using Distributions: Normal, MvNormal`.
# =============================================================================

## ================= Noise models =====================
abstract type NoiseModel end

"No noise."
struct NoNoise <: NoiseModel end

"Independent `Normal(0, σ)` noise on every entry."
struct NaiveNoise{T <: Real} <: NoiseModel
    σ::T
end

"Correlated Gaussian noise (model A): per group covariance σ² R₃, R₃ = (3/2)(I - J/3), restricted to the kept outcomes."
struct IsotropicNoise{T <: Real, M <: MeasurementSet} <: NoiseModel
    σ::T
    ms::M
end

"Zero-mean Gaussian noise with a fixed (possibly singular) K×K covariance Σ."
struct CovariantNoise{T <: Real} <: NoiseModel
    Σ::Matrix{T}
    L::Matrix{T}   # Σ = L L'
end
CovariantNoise(Σ::AbstractMatrix{<:Real}) = CovariantNoise(Matrix(Σ), psd_factor(Σ))

"L with Σ = L L' for a positive semidefinite (possibly singular) Σ."
function psd_factor(Σ)
    F = eigen(Symmetric(Matrix(Σ)))
    tol = sqrt(eps()) * max(maximum(F.values), eps())
    all(F.values .> -tol) || throw(ArgumentError("Σ must be positive semidefinite"))
    keep = F.values .> tol
    return F.vectors[:, keep] .* sqrt.(F.values[keep])'
end

"""
Multinomial shot noise, Gaussian approximation: per group Cov(e | p) = (diag p - p p')/n_s.
`n_s` is a number or a vector with one entry per group (see `groups`).
"""
struct ShotNoise{S <: Union{Real, AbstractVector{<:Real}}, M <: MeasurementSet} <:
       NoiseModel
    n_s::S
    ms::M
end

shots(n::ShotNoise{<:Real}, ::Int) = n.n_s
shots(n::ShotNoise{<:AbstractVector}, g::Int) = n.n_s[g]

## ================= Covariance of one group =====================
"Block σ² R₃ restricted to the kept outcomes (the first `n_outcomes`)."
function group_covariance(n::IsotropicNoise)
    k = 1:n_outcomes(n.ms)
    return n.σ^2 * (1.5I - fill(0.5, 3, 3))[k, k]
end

" Cov(e | p) for the probabilities p of the kept outcomes of group `gi`."
group_covariance(n::ShotNoise, p, gi) = (Diagonal(p) - p * p') / shots(n, gi)

## ================= Sampling =====================
"Noise matrix E with the size of X, columns are independent draws."
noise_sample(::NoNoise, X) = zeros(float(real(eltype(X))), size(X))
noise_sample(n::NaiveNoise, X) = rand(Normal(0, n.σ), size(X))

function noise_sample(n::IsotropicNoise, X)
    size(X, 1) == nrows(n.ms) ||
        throw(DimensionMismatch("X has $(size(X, 1)) rows, expected $(nrows(n.ms))"))
    L = psd_factor(group_covariance(n))
    E = zeros(float(real(eltype(X))), size(X))
    for g in groups(n.ms)
        E[g, :] = L * randn(size(L, 2), size(X, 2))
    end
    return E
end

function noise_sample(n::CovariantNoise, X)
    size(X, 1) == size(n.Σ, 1) ||
        throw(DimensionMismatch("X has $(size(X, 1)) rows, expected $(size(n.Σ, 1))"))
    r = size(n.L, 2)
    return n.L * rand(MvNormal(zeros(r), ones(r)), size(X, 2))
end

function noise_sample(n::ShotNoise, X)
    size(X, 1) == nrows(n.ms) ||
        throw(DimensionMismatch("X has $(size(X, 1)) rows, expected $(nrows(n.ms))"))
    E = zeros(float(real(eltype(X))), size(X))
    for (gi, g) in enumerate(groups(n.ms)), c in axes(X, 2)

        L = psd_factor(group_covariance(n, max.(real.(X[g, c]), 0), gi))
        E[g, c] = L * randn(size(L, 2))
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
    size(S, 1) == nrows(n.ms) ||
        throw(DimensionMismatch("S has $(size(S, 1)) rows, expected $(nrows(n.ms))"))
    Σ = zeros(size(S, 1), size(S, 1))
    for g in groups(n.ms)
        Σ[g, g] = group_covariance(n)
    end
    return Σ
end

function noise_covariance(n::ShotNoise, S, B, b)
    size(S, 1) == nrows(n.ms) ||
        throw(DimensionMismatch("S has $(size(S, 1)) rows, expected $(nrows(n.ms))"))
    SB = to_real.(S * B)
    d = isqrt(size(S, 2))
    p̄ = to_real.(S * vec(I(d))) ./ d
    Σ = zeros(size(S, 1), size(S, 1))
    for (gi, g) in enumerate(groups(n.ms))
        Σ[g, g] = group_covariance(n, p̄[g], gi) - b * SB[g, :] * SB[g, :]' / shots(n, gi)
    end
    return Σ
end

"F = S_B† Σ_E⁺ S_B, the noise-whitened information matrix."
information_matrix(n::NaiveNoise, S, B, b) = (SB = S * B; SB' * SB / n.σ^2)
function information_matrix(n::NoiseModel, S, B, b)
    SB = S * B
    return SB' * pinv(Hermitian(noise_covariance(n, S, B, b))) * SB
end

"Per-target MSE of the optimal linear estimator, diag(P_B† (1/b + F)⁻¹ P_B)."
function mse_theory(S, B, P, b, noise::NoiseModel)
    F = information_matrix(noise, S, B, b)
    T = P' * B
    return to_real.(diag(T * inv(Hermitian(I / b + F)) * T'))
end

"Noisy weights W̃_X = b P_B† S_B† Γ⁺ with Γ = b S_B S_B† + Σ_E."
function W̃X_theory(S, B, P, b, noise::NoiseModel)
    SB = S * B
    Γ = b * SB * SB' + noise_covariance(noise, S, B, b)
    return b * (P' * B) * SB' * pinv(Hermitian(Γ))
end
function mse_theory(S, B, P, b, noise::CovariantNoise)
    SB = S * B
    T = P' * B
    Γ = Hermitian(b * SB * SB' + noise.Σ)
    return to_real.(diag(b * T * T' - b^2 * T * (SB' * pinv(Γ) * SB) * T'))
end

# mse_theory(S, B, P, b, ::NoNoise) = zeros(size(P, 2))
function mse_theory(S, B, P, b, ::NoNoise)
    SB = S * B
    T = P' * B
    return to_real.(diag(b * T * (I - pinv(SB) * SB) * T'))
end

W̃X_theory(S, B, P, b, ::NoNoise) = (P' * B) * pinv(S * B)

"Whitened singular values sqrt(λ_p) of F and overlaps of the targets with its eigenvectors. For NaiveNoise(σ) these are σ_p/σ."
function sv_overlap(S, B, P, b, noise::NoiseModel)
    F = eigen(Hermitian(information_matrix(noise, S, B, b)))
    return (vals = sqrt.(max.(F.values, 0)), overlaps = P' * B * F.vectors)
end
