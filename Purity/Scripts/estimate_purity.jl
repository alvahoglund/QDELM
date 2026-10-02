includet("..\\Core\\estimate_purity.jl")
using JLD2
using OhMyThreads
using CairoMakie
scheduler = DynamicScheduler()
##
S = load("DefaultSystems/scrambling_map_A.jld2", "S")
sys = load("DefaultSystems/scrambling_map_A.jld2", "sys")
measurementset = load("DefaultSystems/scrambling_map_A.jld2", "measurements")
##
nbr_states_train = 10^4
nbr_states_test = 10^4
Ω_train = random_mixed_states(nbr_states_train, sys)
Ω_test = random_mixed_states(nbr_states_test, sys)
X_train = QDELM.get_X(S, Ω_train)
X_test = QDELM.get_X(S, Ω_test)
Y_train = get_purity(Ω_train)
Y_test = get_purity(Ω_test)

##
result = estimate_purity(X_train, Y_train, X_test, Y_test, NoNoise())
println("MSE: ", result.mse)

## Vary noise level
σE_list = logrange(1e-6, 1, length = 20)
noises = [NaiveNoise(σE) for σE in σE_list]
# noises = [IsotropicNoise(σE, measurementset) for σE in σE_list]
mse_list = tmap(noise -> estimate_purity(X_train, Y_train, X_test, Y_test, noise).mse, noises; scheduler)
fig = Figure()
ax = Axis(
    fig[1, 1], xlabel = "Noise level σE", ylabel = "MSE", xscale = log10, yscale = log10)
scatterlines!(ax, σE_list, mse_list, color = :blue)
fig