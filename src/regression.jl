## Datasets

get_X(S, Ω) = to_real.(S * Ω)
get_X_noisy(X, σE) = X + rand(Normal(0, σE), size(X))
get_Y(Σ, Ω) = Σ' * Ω

function center_X(; X_train, X_test)
    XC_train = X_train .- mean(X_train, dims = 2)
    XC_test = X_test .- mean(X_train, dims = 2)
    return (XC_train = XC_train, XC_test = XC_test)
end
add_bias(XC) = vcat(XC, ones(1, size(XC, 2)))
function preprocess_X(; X_train, X_test)
    return (Z_train = add_bias(center_X(X_train = X_train, X_test = X_test).XC_train),
        Z_test = add_bias(center_X(X_train = X_train, X_test = X_test).XC_test))
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

mse(Y_true, Y_pred) = mean((Y_true - Y_pred) .^ 2)

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