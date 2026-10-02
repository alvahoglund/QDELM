using JLD2, CairoMakie
includet("..\\Core\\estimate_spin.jl")
includet("..\\Plots\\vary_data_size.jl")

using OhMyThreads
scheduler = DynamicScheduler()
## Load system
S = load("DefaultSystems/scrambling_map_D.jld2", "S")
sys = load("DefaultSystems/scrambling_map_D.jld2", "sys")
measurementset = load("DefaultSystems/scrambling_map_D.jld2", "measurements")

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

## Noise free, linear scale, vary training size
nbr_train_states_list = [i for i in range(1, 100)]

result_noise_free = tmap(
    nbr_train_states -> fit_and_compare(;
        X_train = datasets.X_train[:, 1:nbr_train_states], X_test = datasets.X_test,
        Y_train = datasets.Y_train[:, 1:nbr_train_states], Y_test = datasets.Y_test, noise = NoNoise(),
        Σ = Σ, S = S, B = B, b = b),
    nbr_train_states_list; scheduler);

mse_against_training_size(
    nbr_train_states_list, mean.(getproperty.(result_noise_free, :mse)),
    mean.(getproperty.(result_noise_free, :mse_diff)), mean.(getproperty.(result_noise_free, :weight_diff)), S, identity)

## Noisy, logscale, vary training size
nbr_train_states_list = unique!(floor.(Int, [i for i in logrange(1, 1e4, length = 50)]))
# noise = NaiveNoise(1e-2)
# noise = IsotropicNoise(1e-2, measurementset)
noise = ShotNoise(1e-2, measurementset)

result_noisy_train = tmap(
    nbr_train_states -> fit_and_compare(;
        X_train = datasets.X_train[:, 1:nbr_train_states], X_test = datasets.X_test,
        Y_train = datasets.Y_train[:, 1:nbr_train_states], Y_test = datasets.Y_test,
        noise = noise, Σ = Σ, S = S, B = B, b = b),
    nbr_train_states_list; scheduler)

fig_noisy_train = mse_against_training_size(
    nbr_train_states_list, mean.(getproperty.(result_noisy_train, :mse)),
    mean.(getproperty.(result_noisy_train, :mse_diff)), mean.(getproperty.(result_noisy_train, :weight_diff)), S; xscale = log10, title = string(typeof(noise)), legend = false)

#save("Figures/vary_training_size_noisy.png", fig_noisy_train)
## Noisy, logscale, vary test size
nbr_test_states_list = unique!(floor.(Int, [i for i in logrange(1, 1e4, length = 50)]))
                                    
result_noisy_test = tmap(
    nbr_test_states -> fit_and_compare(;
        X_train = datasets.X_train, X_test = datasets.X_test[:, 1:nbr_test_states],
        Y_train = datasets.Y_train, Y_test = datasets.Y_test[:, 1:nbr_test_states],
        noise = noise, Σ = Σ, S = S, B = B, b = b),
    nbr_test_states_list; scheduler)

fig_noisy_test = mse_against_test_size(
    nbr_test_states_list, mean.(getproperty.(result_noisy_test, :mse)),
    mean.(getproperty.(result_noisy_test, :mse_diff)), log10)

#save("Figures/vary_test_size_noisy.png", fig_noisy_test)