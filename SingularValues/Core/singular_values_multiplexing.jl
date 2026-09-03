using QDELM, LinearAlgebra, Random

function rand_S(; sys, measurements, t_func = () -> rand(100:200), param_funcs = QDELM.random_param_functions)
    hams = QDELM.matrix_representation_hams(QDELM.hamiltonians(sys.grids, param_funcs()), sys)
    ρ_res = ground_state(hams.res)
    t = t_func()
    return scrambling_map(sys, measurements, ρ_res, hams.total, t)
end

function hamiltonian_multiplexing(
        ; sys, measurements, M_max, param_funcs = QDELM.random_param_functions,
        t_func = () -> rand(100:200), seed = 8310)
    Random.seed!(seed)
    [rand_S(sys = sys, measurements = measurements, t_func = t_func, param_funcs = param_funcs)
     for M in 1:M_max]
end

function time_multiplexing(;
        sys, measurements, M_max, t_func = () -> rand(100:200),
        seed = 7284, param_funcs = QDELM.random_param_functions)
    Random.seed!(seed)
    hams = QDELM.matrix_representation_hams(QDELM.hamiltonians(sys.grids, param_funcs()), sys)
    ρ_res = ground_state(hams.res)
    S_list = [scrambling_map(sys, measurements, ρ_res, hams.total, t_func())
              for M in 1:M_max]
    return S_list
end

function get_svd_vals(S_list, B)
    svd_list = svdvals(vcat(S_list...)*B)
    if length(svd_list) < size(B, 2)
        svd_list = vcat(svd_list, zeros(size(B, 2) - length(svd_list)))
    end
    return svd_list
end

function svd_lists_multiplexing(S_list, B)
    [get_svd_vals(S_list[1:m], B) for m in 1:length(S_list)]
end

κ(σ_max, σ_min) = σ_min > 10^-10 ? σ_max / σ_min : NaN

function rescale_by_m(mat)
    map(m -> mat[:, m] ./ sqrt(m), 1:size(mat, 2)) |>
    rescaled -> hcat(rescaled...)
end