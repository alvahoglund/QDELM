using QDELM, LinearAlgebra, Random

# Set BLAS to single-threaded to avoid oversubscription
BLAS.set_num_threads(1)

function rand_S(; sys, measurements, t_func = () -> rand(100:200), param_funcs = QDELM.random_param_functions())
    hams = QDELM.matrix_representation_hams(QDELM.hamiltonians(sys.grids, param_funcs), sys)
    ρ_res = ground_state(hams.res)
    t = t_func()
    return scrambling_map(sys, measurements, ρ_res, hams.total, t)
end

function hamiltonian_multiplexing(
        ; sys, measurements, M_max, param_funcs = QDELM.random_param_functions(),
        t_func = () -> rand(100:200), seed = 8310)
    Random.seed!(seed)
    [rand_S(sys = sys, measurements = measurements, t_func = t_func, param_funcs = param_funcs)
     for M in 1:M_max]
end

function time_multiplexing(;
        sys, measurements, M_max, t_func = () -> rand(100:200),
        seed = 7284, param_funcs = QDELM.random_param_functions())
    Random.seed!(seed)
    hams = QDELM.matrix_representation_hams(QDELM.hamiltonians(sys.grids, param_funcs), sys)
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

function singular_values(G)
    svd_list = svdvals(G)
    if length(svd_list) < size(G, 1)
        svd_list = vcat(svd_list, zeros(size(G, 1) - length(svd_list)))
    end
    return svd_list
end

function singular_values_fromG(G)
    λ = eigvals(G)
    if length(λ) < size(G, 1)
        λ = vcat(λ, zeros(size(G, 1) - length(λ)))
    end
    return sort(sqrt.(abs.(λ)))
end

## ======================== Ensemble average with varying params=========================
function get_ensemble_average(; nbr_dots_res, qn_res, ham_param_funcs, nbr_samples, t_func)
    grid = QDELM.generate_grid(2, nbr_dots_res)
    sys = tight_binding_system(grid, qn_res)
    measurements = QDELM.charge_probabilities(sys)
    B = QDELM.get_B(sys.Hs_main, sys.H_main)

    D = size(B, 2)
    chunk_size = cld(nbr_samples, Threads.nthreads())
    chunks = Iterators.partition(1:nbr_samples, chunk_size)
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            G_local = zeros(ComplexF64, D, D)
            for _ in chunk
                ham_symb = QDELM.hamiltonians(grid, ham_param_funcs)
                hams_mat = QDELM.matrix_representation_hams(ham_symb, sys)
                ψ_res = ground_state(hams_mat.res)
                SB = scrambling_map(sys, measurements, ψ_res, hams_mat.total, t_func())*B
                G_local += SB' * SB
            end
            G_local
        end
    end
    G_sum = reduce((a, b) -> a .+ b, fetch.(tasks))
    G_avg = G_sum / nbr_samples
    return G_avg
end

function get_ensemble_average_sv(;
        nbr_dots_res, qn_res, ham_param_funcs, nbr_samples, t_func)
    G_avg = get_ensemble_average(
        nbr_dots_res = nbr_dots_res, qn_res = qn_res,
        ham_param_funcs = ham_param_funcs, nbr_samples = nbr_samples,
        t_func = t_func)
    return singular_values_fromG(G_avg)
end

function get_ensemble_average_sv_params(;
        nbr_dots_res, qn_res, ham_param_func_list, nbr_samples, t_func)
    map(
        ham_params -> get_ensemble_average_sv(;
            nbr_dots_res = nbr_dots_res, qn_res = qn_res,
            ham_param_funcs = ham_params, nbr_samples = nbr_samples,
            t_func = t_func),
        ham_param_func_list)
end

function get_ensemble_average_sv_params_settings(;
        settings, ham_param_func_list, nbr_samples, t_func)
    sv_dict = Dict{Tuple{Int, Int}, Vector{Vector{Float64}}}()

    for setting in settings
        sv_dict[(setting[1],
            setting[2])] = get_ensemble_average_sv_params(
            nbr_dots_res = setting[1], qn_res = setting[2],
            ham_param_func_list = ham_param_func_list, nbr_samples = nbr_samples,
            t_func = t_func)
    end
    return sv_dict
end

## ======================== Ensemble average with varying qn =========================

function get_ensemble_average_qn(; nbr_dots_res, ham_param_funcs, nbr_samples, t_func)
    ## Get ensamble average for varying qns. The same Hamiltonians are used for all qns
    grid = QDELM.generate_grid(2, nbr_dots_res)
    measurements = QDELM.charge_probabilities(grid.total)
    n_qn = 2 * nbr_dots_res + 1

    sys_list = map(Base.Fix1(tight_binding_system, grid), 0:(n_qn - 1))
    B = QDELM.get_B(sys_list[1].Hs_main, sys_list[1].H_main)
    m_ops_list = map(
        sys -> QDELM.matrix_representation_ops(measurements, sys.H_total), sys_list)

    D = size(B, 2)
    chunk_size = cld(nbr_samples, Threads.nthreads())
    chunks = Iterators.partition(1:nbr_samples, chunk_size)
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            G_local = [zeros(ComplexF64, D, D) for _ in 1:n_qn]
            for _ in chunk
                ham_symb = QDELM.hamiltonians(grid, ham_param_funcs)
                for qn_res_idx in 1:n_qn
                    hams_mat = QDELM.matrix_representation_hams(
                        ham_symb, sys_list[qn_res_idx])
                    ψ_res = ground_state(hams_mat.res)
                    SB = scrambling_map(
                        sys_list[qn_res_idx], m_ops_list[qn_res_idx], ψ_res,
                        hams_mat.total, t_func())*B
                    G_local[qn_res_idx] += SB' * SB
                end
            end
            G_local
        end
    end
    G_sum = reduce((a, b) -> a .+ b, fetch.(tasks))
    G_avg = map(G -> G / nbr_samples, G_sum)
    return G_avg
end

function get_ensemble_average_res_qn(; nbr_dots_res_list, ham_param_funcs, nbr_samples, t_func)
    # Construct a dictionary to hold the ensemble average for each nbr_dots_res
    ens_avg_dict = Dict{Int, Vector{Matrix{ComplexF64}}}()
    for nbr_dots_res in nbr_dots_res_list
        ens_avg_dict[nbr_dots_res] = get_ensemble_average_qn(
            nbr_dots_res = nbr_dots_res, ham_param_funcs = ham_param_funcs,
            nbr_samples = nbr_samples, t_func = t_func)
    end
    return ens_avg_dict
end

function get_ensemble_average_sv_qn(; nbr_dots_res, ham_param_funcs, nbr_samples, t_func)
    G_avg = get_ensemble_average_qn(
        nbr_dots_res = nbr_dots_res, ham_param_funcs = ham_param_funcs,
        nbr_samples = nbr_samples, t_func = t_func)
    return [singular_values_fromG(G) for G in G_avg]
end

function get_ensemble_average_sv_res_qn(; nbr_dots_res_list, ham_param_funcs, nbr_samples, t_func)
    sv_dict = Dict{Int, Vector{Vector{Float64}}}()
    for nbr_dots_res in nbr_dots_res_list
        sv_dict[nbr_dots_res] = get_ensemble_average_sv_qn(
            nbr_dots_res = nbr_dots_res, ham_param_funcs = ham_param_funcs,
            nbr_samples = nbr_samples, t_func = t_func)
    end
    return sv_dict
end
