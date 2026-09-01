using JLD2, QDELM, GLMakie
includet("..\\Core\\entanglement_detection.jl")
includet("..\\Plots\\entanglement_detection.jl")
## ================ System ======================
S = load("DefaultSystems/scrambling_map_D.jld2", "S")
sys = load("DefaultSystems/scrambling_map_D.jld2", "sys")

## ================ Datasets ======================
nbr_ent_states_train = 10^5
nbr_ent_states_test = 10^5
nbr_sep_states_train = 10^5
nbr_sep_states_test = 10^5
state_names = [QDELM.singlet]

(Ω_ent_train,
    Ω_sep_train,
    Ω_train,
    Y_train) = get_Ω_Y(
    nbr_ent_states_train, nbr_sep_states_train, sys.Hs_main, sys.H_main, state_names)

(Ω_ent_test,
    Ω_sep_test,
    Ω_test,
    Y_test) = get_Ω_Y(
    nbr_ent_states_test, nbr_sep_states_test, sys.Hs_main, sys.H_main, state_names)

X_train = get_X(S, Ω_train)
X_test = get_X(S, Ω_test)

σE = 0
lew = contruct_EW(
    X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test, σE = σE)

## ================ Plot db ======================
(Ω_sub_ent,
    Ω_sub_sep) = project_on_sub_spin_basis(Ω_ent_test, Ω_sep_test, sys.Hs_main, sys.H_main)
W_eff_sub = effective_weight_matrix(
    lew.W, sys.Hs_main, sys.H_main, S, Ω_train, sub_space = true)'

fig = Figure()
plot_linear_ew!(fig[1, 1], W_eff_sub, Ω_sub_sep, Ω_sub_ent;
    step_size_sep = 1, step_size_ent = 10^3, title = "Linear Decision Boundary")
fig

## ================ Plot heatmap of the effective weight matrix
W_eff = effective_weight_matrix(
    lew.W, sys.Hs_main, sys.H_main, S, Ω_train, sub_space = false)'
fig = Figure()
plot_heatmap_W_spin_basis!(fig[1, 1], W_eff)
fig

## =============== Plot heatmap with different noise levels ======================lew = contruct_EW(
lew0 = contruct_EW(
    X_train = X_train, Y_train = Y_train, X_test = X_test,
    Y_test = Y_test, σE = 0)
Weff0 = effective_weight_matrix(
    lew0.W, sys.Hs_main, sys.H_main, S, Ω_train, sub_space = false)'
lew2 = contruct_EW(
    X_train = X_train, Y_train = Y_train, X_test = X_test,
    Y_test = Y_test, σE = 10^-2)
Weff2 = effective_weight_matrix(
    lew2.W, sys.Hs_main, sys.H_main, S, Ω_train, sub_space = false)'

fig = Figure()
plot_heatmap_W_spin_basis!(fig[1, 1], Weff0)
plot_heatmap_W_spin_basis!(fig[1, 2], Weff2)
fig