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
targets = [(:σ0, :σz), (:σx, :σz)]
targets_idx = [Pm_dict[t] for t in targets]

Ω_train = randomize_hs_states(sys, nbr_states_training)
Ω_test = randomize_hs_states(sys, nbr_states_test)
Σ = Pm[:, targets_idx]

datasets = (X_train = get_X(S, Ω_train),
    X_test = get_X(S, Ω_test),
    Y_train = get_Y(Σ, Ω_train),
    Y_test = get_Y(Σ, Ω_test))
## 
σE_list = logrange(1e-7, 1, length = 30)
# noises = [NaiveNoise(σE) for σE in σE_list]
# noises = [IsotropicNoise(σE, measurementset) for σE in σE_list]
# noises = [ShotNoise(1/σE, measurementset) for σE in σE_list]
results = map(noise -> fit_and_evaluate(; datasets..., noise), noises)
mse_list = map(r -> r.mse, results)
mse_theory_list = map(noise -> mse_theory(S, B, Σ, b, noise), noises)

# σ = 1 gives the plain singular values of S*B, so the plot's √b·σ_p lines are in σE units
sv_overlaps = sv_overlap(S, B, Σ, b, NaiveNoise(1.0))

mse_against_noise(mse_list, mse_theory_list, targets, sv_overlaps, b)