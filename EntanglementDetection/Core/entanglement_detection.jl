using QDELM
using LinearAlgebra, Statistics, Distributions, Random

## Datasets
function get_prod_states(nbr_sep_states, Hs_main, H_main)
    stack(vec(density_matrix(QDELM.random_product_state(Hs_main, H_main)))
    for i in 1:nbr_sep_states)
end

function get_sep_states(nbr_sep_states, Hs_main, H_main)
    stack(vec(density_matrix(QDELM.random_separable_state(
              rand(1:3), Hs_main, H_main)))
    for i in 1:nbr_sep_states)
end

function get_sep_states(nbr_sep_states, rank, Hs_main, H_main)
    stack(vec(density_matrix(QDELM.random_separable_state(
              rank, Hs_main, H_main)))
    for i in 1:nbr_sep_states)
end

function get_ent_states(nbr_ent_states, H_main, state_names)
    #Genereate equal number of entangled states for each state type, 
    #so that the total number of entangled states can be less than nbr_ent_states
    p_list = range(0, 1 / 3, length = nbr_ent_states ÷ length(state_names))
    mapreduce(state -> stack(vec(QDELM.werner_state(state, p, H_main)) for p in p_list),
        hcat, state_names)
end

function get_Ω_Y(nbr_ent_states, nbr_sep_states, Hs_main, H_main, state_names)
    Ω_ent = get_ent_states(nbr_ent_states, H_main, state_names)
    Ω_sep = get_sep_states(nbr_sep_states, Hs_main, H_main)
    Ω = hcat(Ω_ent, Ω_sep)
    Y = hcat((-1) * ones(1, nbr_ent_states), ones(1, nbr_sep_states))
    perm = randperm(size(Ω, 2))
    Ω = Ω[:, perm]
    Y = Y[1:1, perm]
    return Ω_ent, Ω_sep, Ω, Y
end

## Model fitting and evaluation
get_accuracy(Y_true, Y_pred) = mean((Y_true .> 0) .== (Y_pred .> 0))
function get_accuracy_classes(Y_true, Y_pred)
    sep_mask = Y_true .== 1
    ent_mask = Y_true .== -1
    Dict(
        "Separable" => mean(Y_pred[sep_mask] .> 0),
        "Entangled" => mean(Y_pred[ent_mask] .< 0)
    )
end

function contruct_EW(; X_train, Y_train, X_test, Y_test, σE::Number,
        feature_transformation_alg = QDELM.IdentityFeatureTransformation())
    X̃_train = add_noise(X_train, σE)
    X̃_test = add_noise(X_test, σE)
    Z_train, Z_test = preprocess_X(X_train = X̃_train, X_test = X̃_test)
    Z_train_poly = feature_transformation(Z_train, feature_transformation_alg)
    Z_test_poly = feature_transformation(Z_test, feature_transformation_alg)
    W = regression(Z_train_poly, Y_train)
    Y_pred = W * Z_test_poly
    return (W = W, Y_pred = Y_pred, Y_test = Y_test,
        Z_train_poly = Z_train_poly, Z_test_poly = Z_test_poly)
end

function test_EW(; X_train, Y_train, X_test, Y_test, σE::Number,
        feature_transformation_alg = QDELM.IdentityFeatureTransformation())
    W, Y_pred,
    Y_test = contruct_EW(X_train = X_train, Y_train = Y_train,
        X_test = X_test, Y_test = Y_test, σE = σE,
        feature_transformation_alg = feature_transformation_alg)
    return get_accuracy(Y_test, Y_pred)
end

function get_sub_spin_basis(Hs_main, H_main)
    Pm, Pm_dict = pauli_matrix(Hs_main, H_main)
    Pm[:,
        [Pm_dict[(:σ0, :σ0)], Pm_dict[(:σx, :σx)],
            Pm_dict[(:σy, :σy)], Pm_dict[(:σz, :σz)]]]
end

function project_on_sub_spin_basis(Ω_ent, Ω_sep, Hs_main, H_main)
    Pm_sub = get_sub_spin_basis(Hs_main, H_main)
    Ω_sub_ent = to_real.(Ω_ent' * Pm_sub)
    Ω_sub_sep = to_real.(Ω_sep' * Pm_sub)
    return Ω_sub_ent, Ω_sub_sep
end

function effective_weight_matrix(W, Hs_main, H_main, S, Ω_train; sub_space = false)
    Pm, Pm_dict = pauli_matrix(Hs_main, H_main)
    if sub_space
        Pm = get_sub_spin_basis(Hs_main, H_main)
    end
    Σ00 = Pm[1:end, Pm_dict[(:σ0, :σ0)]]
    WX = W[1:1, 1:(end - 1)]
    1/4 * WX * S * (Pm - mean(Ω_train, dims = 2) * Σ00' * Pm)
end

function effective_weight_matrix_sub(W, Hs_main, H_main, S, Ω_train)
    effective_weight_matrix(W, Hs_main, H_main, S, Ω_train) *
    get_sub_spin_basis(Hs_main, H_main)
end

