using JLD2, QDELM, GLMakie
includet("..\\Core\\entanglement_detection.jl")
includet("..\\Plots\\entanglement_detection.jl")

## ================ System ======================
S = load("DefaultSystems/scrambling_map_D.jld2", "S")
sys = load("DefaultSystems/scrambling_map_D.jld2", "sys")

## ================ Datasets ======================

nbr_ent_states_train = 10^4
nbr_ent_states_test = 10^4
nbr_sep_states_train = 10^4
nbr_sep_states_test = 10^4
state_names = [QDELM.singlet, QDELM.triplet_0, QDELM.triplet_plus, QDELM.triplet_minus]

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
new = contruct_EW(
    X_train = X_train, Y_train = Y_train, X_test = X_test, Y_test = Y_test, σE = σE,
    feature_transformation_alg = QDELM.Polynomial2FeatureTransformation())

##
(Ω_sub_ent,
    Ω_sub_sep) = project_on_sub_spin_basis(Ω_ent_test, Ω_sep_test, sys.Hs_main, sys.H_main)

plot_nonlinear_db_spin_space(
    Ω_sub_sep, Ω_sub_ent, new.W, S, vec(mean(X_train, dims = 2)),
    QDELM.Polynomial2FeatureTransformation())

accuracy = get_accuracy_classes(new.Y_test, new.Y_pred)