## Datasets

get_X(S, Ω) = to_real.(S * Ω)
get_Y(Σ, Ω) = Σ' * Ω

function center_X(; X_train, X_test)
    XC_train = X_train .- mean(X_train, dims = 2)
    XC_test = X_test .- mean(X_train, dims = 2)
    return (XC_train = XC_train, XC_test = XC_test)
end
add_bias(XC) = vcat(XC, ones(1, size(XC, 2)))
function preprocess_X(; X_train, X_test)
    cX = center_X(X_train = X_train, X_test = X_test)
    return (Z_train = add_bias(cX.XC_train), Z_test = add_bias(cX.XC_test))
end

"""
    add_noise_center_bias(; X_train, X_test, noise)

Performs the equivalent of:
    X_train_noisy = add_noise(X_train, noise)
    X_test_noisy  = add_noise(X_test, noise)
    preprocess_X(X_train = X_train_noisy, X_test = X_test_noisy)
"""
function add_noise_center_bias(; X_train, X_test, noise::NoiseModel)
    d, n_train, n_test = size(X_train, 1), size(X_train, 2), size(X_test, 2)
    
    # 1. Allocate final matrices and fill with (X + noise) + bias row
    Z_train = Matrix{eltype(X_train)}(undef, d + 1, n_train)
    Z_test  = Matrix{eltype(X_test)}(undef, d + 1, n_test)
    @views Z_train[1:d, :] .= noise isa NoNoise ? X_train : add_noise(X_train, noise)
    @views Z_test[1:d, :]  .= noise isa NoNoise ? X_test : add_noise(X_test, noise)
    fill!(@view(Z_train[d + 1, :]), 1)
    fill!(@view(Z_test[d + 1, :]), 1)

    # 2. Mean of noisy training rows, then center both in-place via views
    μ = mean(@view(Z_train[1:d, :]), dims = 2)
    @views Z_train[1:d, :] .-= μ
    @views Z_test[1:d, :]  .-= μ

    return (; Z_train, Z_test)
end

function split_train_test(X, Y, train_fraction = 0.5)
    nbr_train = round(Int, size(X, 1) * train_fraction)
    X_train, X_test = X[1:nbr_train, :], X[(nbr_train + 1):end, :]
    Y_train, Y_test = Y[1:nbr_train, :], Y[(nbr_train + 1):end, :]
    return X_train, X_test, Y_train, Y_test
end

##
function regression(X_train, Y_train)
    return Y_train * pinv(X_train)
end

# mse(Y_true, Y_pred) = mean((Y_true - Y_pred) .^ 2)
function mse(Y_true::AbstractArray, Y_pred::AbstractArray)
    @assert size(Y_true) == size(Y_pred)
    s = mapreduce((yt, yp) -> abs2(yt - yp), +, Y_true, Y_pred)
    return s / length(Y_true)
end

## Feature transformation for nonlinear regression
abstract type FeatureTransformation end
struct Polynomial2FeatureTransformation <: FeatureTransformation end

struct Polynomial2SectionFeatureTransformation <: FeatureTransformation
    section_size::Int
end

struct IdentityFeatureTransformation <: FeatureTransformation end

function degree_2_polynomial_feature_transformation(X)
    n_features, n_samples = size(X)
    vcat(X, X .^ 2,
        [X[i:i, :] .* X[j:j, :] for i in 1:n_features for j in (i + 1):n_features]...)
end

function feature_transformation(X, alg::Polynomial2FeatureTransformation)
    degree_2_polynomial_feature_transformation(X)
end

function feature_transformation(X, alg::Polynomial2SectionFeatureTransformation)
    #Split the input data into sections to reduce the number of features after transformation
    n_features, n_samples = size(X)
    n_sections = ceil(Int, n_features / alg.section_size)
    X_sections = [X[((i - 1) * alg.section_size + 1):min(
                      i * alg.section_size, n_features), :]
                  for i in 1:n_sections]
    transformed_sections = [degree_2_polynomial_feature_transformation(X_sec)
                            for X_sec in X_sections]
    vcat(transformed_sections...)
end

function feature_transformation(X, alg::IdentityFeatureTransformation)
    X
end