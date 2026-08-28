function regression(X_train, Y_train)
    return Y_train * pinv(X_train)
end

function split_train_test(X, Y, train_fraction = 0.5)
    nbr_train = round(Int, size(X, 1) * train_fraction)
    X_train, X_test = X[1:nbr_train, :], X[(nbr_train + 1):end, :]
    Y_train, Y_test = Y[1:nbr_train, :], Y[(nbr_train + 1):end, :]
    return X_train, X_test, Y_train, Y_test
end

mse(Y_true, Y_pred) = mean((Y_true - Y_pred) .^ 2)

function mse_prediction(S_SVD, Pm, σE, b)
    real.(diag(Pm' * S_SVD.V * diagm((b * σE^2) ./ (b .* S_SVD.S .^ 2 .+ σE^2)) * S_SVD.V' *
               Pm))
end

## Feature transformation for nonlinear regression
abstract type FeatureTransformation end
struct Polynomial2FeatureTransformation <: FeatureTransformation end

struct Polynomial2SectionFeatureTransformation <: FeatureTransformation
    section_size::Int
end

struct IdentityFeatureTransformation <: FeatureTransformation end

function degree_2_polynomial_feature_transformation(X)
    n_samples, n_features = size(X)
    hcat(X, X .^ 2,
        [X[:, i] .* X[:, j] for i in 1:n_features for j in (i + 1):n_features]...)
end

function feature_transformation(X, alg::Polynomial2FeatureTransformation)
    degree_2_polynomial_feature_transformation(X)
end

function feature_transformation(X, alg::Polynomial2SectionFeatureTransformation)
    #Split the input data into sections to reduce the number of features after transformation
    n_samples, n_features = size(X)
    n_sections = ceil(Int, n_features / alg.section_size)
    X_sections = [X[:,
                      ((i - 1) * alg.section_size + 1):min(
                          i * alg.section_size, n_features)]
                  for i in 1:n_sections]
    transformed_sections = [degree_2_polynomial_feature_transformation(X_sec)
                            for X_sec in X_sections]
    hcat(transformed_sections...)
end

function feature_transformation(X, alg::IdentityFeatureTransformation)
    X
end