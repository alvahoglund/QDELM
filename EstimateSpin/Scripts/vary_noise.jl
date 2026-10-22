includet("..\\Core\\estimate_spin.jl")
includet("..\\Plots\\vary_noise.jl")
using JLD2, CairoMakie
## Load system
S = load("DefaultSystems/scrambling_map_D.jld2", "S")
sys = load("DefaultSystems/scrambling_map_D.jld2", "sys")
measurementset = load("DefaultSystems/scrambling_map_D.jld2", "measurements")

Pm, Pm_dict = QDELM.pauli_matrix(sys.Hs_main, sys.H_main)
B = 1/2 .* Pm[:, 2:end]
b = 0.0147
## Generate datasets

nbr_states_training = 10^4
nbr_states_test = 10^4
targets = [(:σ0, :σy), (:σy, :σy)]
targets_idx = [Pm_dict[t] for t in targets]

Ω_train = randomize_hs_states(sys, nbr_states_training)
Ω_test = randomize_hs_states(sys, nbr_states_test)
P = Pm[:, targets_idx]

datasets = (X_train = get_X(S, Ω_train),
    X_test = get_X(S, Ω_test),
    Y_train = get_Y(P, Ω_train),
    Y_test = get_Y(P, Ω_test))
## 
σE_list = logrange(1e-6, 1, length = 30)
# noises = [NaiveNoise(σE) for σE in σE_list]
#noises = [IsotropicNoise(σE, measurementset) for σE in σE_list]
noises = [ShotNoise(1/σE^2, measurementset) for σE in σE_list]
results = map(noise -> fit_and_evaluate(; datasets..., noise), noises)
mse_list = map(r -> r.mse, results)
mse_theory_list = map(noise -> mse_theory(S, B, P, b, noise), noises)

sv_overlaps = sv_overlap(S, B, P, b, ShotNoise(1.0, measurementset))

fig = mse_against_noise(mse_list, mse_theory_list, targets, sv_overlaps, b)

save("Figures\\mse_against_noise_res3_shot.png", fig)