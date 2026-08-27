using QDELM, Random, Statistics, LinearAlgebra, Distributions

nbr_dots_main = 2
nbr_dots_res = 3
nbr_electrons_res = 2

sys = tight_binding_system(nbr_dots_main, nbr_dots_res, nbr_electrons_res)
seed = 1234
Random.seed!(seed)
hams = matrix_representation_hams(hamiltonians(sys.grids), sys)
t = [100, 200]

measurements = matrix_representation_ops(charge_probabilities(sys.grids.total), sys.H_total)
ψ_res = ground_state(hams.res)
S = scrambling_map(sys, measurements, ψ_res, hams.total, t)

pm, pm_dict = pauli_matrix(sys.Hs_main, sys.H_main)
Σzz = pm[:, pm_dict[(:σz, :σz)]]
B = 1/2 * pm[:, 2:end]

## ================ Dataset generation ==========================
nbr_states_train = 10^4
nbr_states_test = 10^4
σE = 0.01

Ω_train = stack(vec(QDELM.hilbert_schmidt_ensemble(sys.H_main)) for _ in 1:nbr_states_train)
Ω_test = stack(vec(QDELM.hilbert_schmidt_ensemble(sys.H_main)) for _ in 1:nbr_states_test)

X_train = S * Ω_train
X_test = S * Ω_test

mean_vals = mean(X_train, dims = 2)
XC_train = X_train .- mean_vals
XC_test = X_test .- mean_vals

Z_train = vcat(XC_train, ones(1, nbr_states_train))
Z_test = vcat(XC_test, ones(1, nbr_states_test))

Σzz = vec(ps[(:σz, :σz)])
Y_train = Σzz' * Ω_train
Y_test = Σzz' * Ω_test

## ================= Training without noise and with centering ==========================
W = Y_train * pinv(Z_train)
WX_theory = Σzz' * B * pinv(S*B)
XI_theory = Σzz' * mean(Ω_train, dims = 2)
W_theory = hcat(WX_theory, XI_theory)

weight_diff = norm(W - W_theory) / norm(W_theory)
Y_train_pred = W * Z_train
Y_test_pred = W * Z_test
mse_train = mean((Y_train_pred .- Y_train) .^ 2)
mse = mean((Y_test_pred .- Y_test) .^ 2)

print("Training without noise and with centering:\n")
print("Weight difference: ", weight_diff, "\n")
print("MSE (training): ", mse_train, "\n")
print("MSE (test): ", mse, "\n")

## ================= Training without noise and without centering ==========================

W = Y_train * pinv(X_train)
W_theory = Σzz' * pinv(S)
weight_diff_no_centering = norm(W - W_theory) / norm(W_theory)
Y_train_pred = W * X_train
Y_test_pred = W * X_test
mse_train = mean((Y_train_pred .- Y_train) .^ 2)
mse = mean((Y_test_pred .- Y_test) .^ 2)

print("\nTraining without noise and without centering:\n")
print("Weight difference: ", weight_diff_no_centering, "\n")
print("MSE (training): ", mse_train, "\n")
print("MSE (test): ", mse, "\n")

## ================= Dataset generation with noise ==========================
σE = 10^-5
E_train = rand(Normal(0, σE), size(X_train))
E_test = rand(Normal(0, σE), size(X_test))

X̃_train = X_train .+ E_train
X̃_test = X_test .+ E_test

mean_vals = mean(X̃_train, dims = 2)
X̃C_train = X̃_train .- mean_vals
X̃C_test = X̃_test .- mean_vals

Z̃_train = vcat(X̃C_train, ones(1, nbr_states_train))
Z̃_test = vcat(X̃C_test, ones(1, nbr_states_test))

b = 0.0147

## Training with noise and with centering
W̃ = Y_train * pinv(Z̃_train)
U, D, V = svd(S*B)

A = U * diagm((b * D .^ 2) ./ (b .* D .^ 2 .+ σE^2)) * U'
WX_theory = Σzz' * B * pinv(S*B)
W̃X_theory = WX_theory * A
W̃I_theory = Σzz' * mean(Ω_train, dims = 2)
W̃_theory = hcat(W̃X_theory, W̃I_theory)

weight_diff_no_centering = norm(W̃ - W̃_theory) / norm(W̃_theory)

Y_test_pred = W̃ * Z̃_test
Y_train_pred = W̃ * Z̃_train
mse_train = mean((Y_train_pred .- Y_train) .^ 2)
mse = mean((Y_test_pred .- Y_test) .^ 2)
mse_pred = Σzz' * B * V * diagm((b * σE .^ 2) ./ (b .* D .^ 2 .+ σE^2)) * V' * B' * Σzz

print("\nTraining with noise and with centering:\n")
print("Weight difference: ", weight_diff_no_centering, "\n")
print("MSE (training): ", mse_train, "\n")
print("MSE (test): ", mse, "\n")
print("(MSE_test - MSE_test_pred)/MSE_test_pred: ", (mse - mse_pred)/mse_pred, "\n")