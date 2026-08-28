includet("..\\Core\\estimate_spin.jl")
includet("..\\Plots\\vary_training_data.jl")
using JLD2, CairoMakie
## Load system
S = load("DefaultSystems/scrambling_map_B.jld2", "S")
sys = load("DefaultSystems/scrambling_map_B.jld2", "sys")

Pm, Pm_dict = QDELM.pauli_matrix(sys.Hs_main, sys.H_main)
B = 1/2 .* Pm[:, 2:end]
b = 0.0147

## Generate datasets
nbr_states_training = 10^5
nbr_states_test = 10^5

#targets = [(:σ0, :σz), (:σx, :σz)]
#targets_idx = [Pm_dict[t] for t in targets]
#Σ = Pm[:, targets_idx]
Σ = Pm[:, 2:end] # Include all targets

Ω_train = randomize_hs_states(sys, nbr_states_training)
Ω_test = randomize_hs_states(sys, nbr_states_test)

datasets = (X_train = get_X(S, Ω_train),
    X_test = get_X(S, Ω_test),
    Y_train = get_Y(Σ, Ω_train),
    Y_test = get_Y(Σ, Ω_test))

## Noise free, linear scale
nbr_train_states_list = [i for i in range(1, 100)]

result = map(
    nbr_train_states -> fit_and_compare(;
        X_train = datasets.X_train[:, 1:nbr_train_states], X_test = datasets.X_test,
        Y_train = datasets.Y_train[:, 1:nbr_train_states], Y_test = datasets.Y_test, σE = 0,
        Σ = Σ, S = S, B = B, b = b),
    nbr_train_states_list)

mse_against_training_size(nbr_train_states_list, mean.(getproperty.(result, :mse)),
    mean.(getproperty.(result, :mse_diff)), mean.(getproperty.(result, :weight_diff)), identity)

## Noisy, logscale
nbr_train_states_list = floor.(Int, [i
                                     for i in exp10.(range(log10(1), log10(10^5), length = 100))])
σE = 1e-2

result = map(
    nbr_train_states -> fit_and_compare(;
        X_train = datasets.X_train[:, 1:nbr_train_states], X_test = datasets.X_test,
        Y_train = datasets.Y_train[:, 1:nbr_train_states], Y_test = datasets.Y_test,
        σE = σE, Σ = Σ, S = S, B = B, b = b),
    nbr_train_states_list)

mse_against_training_size(nbr_train_states_list, mean.(getproperty.(result, :mse)),
    mean.(getproperty.(result, :mse_diff)), mean.(getproperty.(result, :weight_diff)), log10)

## Noisy, linear scale
nbr_train_states_list = floor.(Int, [i
                                     for i in range(1, 10^3, length = 100)])

σE = 1e-2

result = map(
    nbr_train_states -> fit_and_compare(;
        X_train = datasets.X_train[:, 1:nbr_train_states], X_test = datasets.X_test,
        Y_train = datasets.Y_train[:, 1:nbr_train_states], Y_test = datasets.Y_test,
        σE = σE, Σ = Σ, S = S, B = B, b = b),
    nbr_train_states_list)

mse_against_training_size(nbr_train_states_list, mean.(getproperty.(result, :mse)),
    mean.(getproperty.(result, :mse_diff)), mean.(getproperty.(result, :weight_diff)), identity)