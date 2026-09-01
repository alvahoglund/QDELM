using QDELM
using LinearAlgebra, Statistics, Distributions

## ================= Datasets ======================
function randomize_hs_states(sys, nbr_states)
    stack(vec(QDELM.hilbert_schmidt_ensemble(sys.H_main)) for _ in 1:nbr_states)
end

## ================= Model fitting and evaluation ======================
get_mse(Y_test, Y_pred) = to_real.(vec(mean((Y_test - Y_pred) .^ 2, dims = 2)))

function fit_and_evaluate_from_states(; Ω_train, Ω_test, S, Σ, σE)
    fit_and_evaluate(
        X_train = get_X(S, Ω_train), X_test = get_X(S, Ω_test), Y_train = get_Y(Σ, Ω_train),
        Y_test = get_Y(Σ, Ω_test), σE = σE)
end

function fit_and_evaluate(; X_train, X_test, Y_train, Y_test, σE)
    X̃_train = get_X_noisy(X_train, σE)
    X̃_test = get_X_noisy(X_test, σE)
    Z_train, Z_test = preprocess_X(X_train = X̃_train, X_test = X̃_test)
    W = QDELM.regression(Z_train, Y_train)
    Y_pred = W * Z_test
    Y_pred_train = W * Z_train
    mse = get_mse(Y_test, Y_pred)
    mse_train = get_mse(Y_train, Y_pred_train)
    return (; mse, mse_train, W)
end

function fit_and_compare(; X_train, X_test, Y_train, Y_test, σE, Σ, S, B, b)
    result = fit_and_evaluate(X_train = X_train, X_test = X_test, Y_train = Y_train,
        Y_test = Y_test, σE = σE)
    mse_theory_val = mse_theory(S, B, Σ, σE, b)
    W_theory = W̃X_theory(S, B, Σ, σE, b)
    weight_diff = norm(result.W[:, 1:(end - 1)] - W_theory) / norm(W_theory)
    mse_diff = norm(result.mse - mse_theory_val) / norm(mse_theory_val)
    mse_diff_train = norm(result.mse_train - mse_theory_val) / norm(mse_theory_val)
    return (; result.mse, result.W, weight_diff, mse_diff, result.mse_train, mse_diff_train)
end

## ================= Theoretical weights ======================
A(U, D, b, σE) = U * diagm((b * D .^ 2) ./ (b .* D .^ 2 .+ σE^2)) * U'
WX_theory(S, B, Σ) = Σ' * B * pinv(S*B)
W̃X_theory(S, B, Σ, σE, b) = WX_theory(S, B, Σ) * A(svd(S*B).U, svd(S*B).S, b, σE)

function mse_theory(S, B, Σ, σE, b)
    U, D, V = svd(S*B)
    return to_real.(diag(Σ' * B * V * diagm((b * σE^2) ./ (b .* D .^ 2 .+ σE^2)) *
                         V' * B' *
                         Σ))
end

sv_overlap(S, B, Σ) = (vals = svd(S*B).S, overlaps = Σ' * B * svd(S*B).V)