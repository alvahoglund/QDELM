using QDELM
using LinearAlgebra, Statistics, Distributions

## ================= Datasets ======================
function randomize_hs_states(sys, nbr_states)
    stack(vec(QDELM.hilbert_schmidt_ensemble(sys.H_main)) for _ in 1:nbr_states)
end

## ================= Model fitting and evaluation ======================
# get_mse(Y_test, Y_pred) = to_real.(vec(mean((Y_test .- Y_pred) .^ 2, dims = 2)))
function get_mse(Y_test::AbstractMatrix{T1}, Y_pred::AbstractMatrix{T2}) where {T1, T2}
    nrows, ncols = size(Y_test)
    T = promote_type(T1, T2)
    out = Vector{real(T)}(undef, nrows)
    for i in 1:nrows
        s = zero(T)
        for j in 1:ncols
            diff = Y_test[i, j] - Y_pred[i, j]
            s += abs2(diff)
        end
        out[i] = s / ncols
    end
    return out
end
function fit_and_evaluate_from_states(; Ω_train, Ω_test, S, P, noise)
    fit_and_evaluate(
        X_train = get_X(S, Ω_train), X_test = get_X(S, Ω_test), Y_train = get_Y(P, Ω_train),
        Y_test = get_Y(P, Ω_test), noise = noise)
end

function fit_and_evaluate(; X_train, X_test, Y_train, Y_test, noise)
    (; Z_train, Z_test) = QDELM.add_noise_center_bias(X_train = X_train, X_test = X_test, noise = noise)
    W = QDELM.regression(Z_train, Y_train)
    Y_pred = W * Z_test
    Y_pred_train = W * Z_train
    mse = get_mse(Y_test, Y_pred)
    mse_train = get_mse(Y_train, Y_pred_train)
    return (; mse, mse_train, W)
end

function fit_and_compare(; X_train, X_test, Y_train, Y_test, noise, P, S, B, b)
    result = fit_and_evaluate(X_train = X_train, X_test = X_test, Y_train = Y_train,
        Y_test = Y_test, noise = noise)
    mse_theory_val = mse_theory(S, B, P, b, noise)
    W_theory = W̃X_theory(S, B, P, b, noise)
    weight_diff = norm(result.W[:, 1:(end - 1)] - W_theory) / norm(W_theory)
    mse_diff = norm(result.mse - mse_theory_val) / norm(mse_theory_val)
    mse_diff_train = norm(result.mse_train - mse_theory_val) / norm(mse_theory_val)
    return (; result.mse, result.W, weight_diff, mse_diff, result.mse_train, mse_diff_train)
end
