using QDELM
using LinearAlgebra, Statistics, Distributions, Random
##
get_purity(Ω) = [real(dot(Ω[:, i], Ω[:, i])) for i in eachindex(Ω[1, :])]

function random_mixed_states(nbr_states, sys)
    p_list = rand(nbr_states)
    mapreduce(
        i -> vec((1 - p_list[i]) * density_matrix(QDELM.random_state(sys.H_main)) +
                 p_list[i] * QDELM.max_mixed_state(sys.H_main)),
        hcat, 1:nbr_states)
end

function get_purity_mse(X, Y, σE)
    W, Y_pred, Y_true = predict_purity(X, Y, σE)
    return QDELM.mse(Y_true, Y_pred)
end

get_purity(Ω) = [real(dot(Ω[:, i], Ω[:, i])) for i in eachindex(Ω[1, :])]

function estimate_purity(X_train, Y_train, X_test, Y_test, σE)
    X̃_train = QDELM.add_noise(X_train, σE)
    X̃_test = QDELM.add_noise(X_test, σE)
    Z_train, Z_test = preprocess_X(; X_train = X̃_train, X_test = X̃_test)
    Z_train_poly = QDELM.feature_transformation(Z_train, QDELM.Polynomial2FeatureTransformation())
    Z_test_poly = QDELM.feature_transformation(Z_test, QDELM.Polynomial2FeatureTransformation())
    W = QDELM.regression(Z_train_poly, reshape(Y_train, 1, length(Y_train)))
    Y_test_pred = W * Z_test_poly
    mse = QDELM.mse(Y_test, vec(Y_test_pred))
    return (W = W, Y_test_pred = Y_test_pred, Y_test = Y_test,
        Z_train_poly = Z_train_poly, Z_test_poly = Z_test_poly, mse = mse)
end