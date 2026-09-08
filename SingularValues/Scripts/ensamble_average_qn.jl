includet("..\\Core\\singular_values_multiplexing.jl")
includet("..\\Plots\\singular_values_multiplexing.jl")
using CairoMakie

ham_param_funcs = QDELM.random_param_functions(t_so = 10)

nbr_samples = 100
t_func = () -> 100
nbr_dots_res_list = [1, 2, 3]

ens_avg_qn = get_ensemble_average_qn(; nbr_dots_res, ham_param_funcs, nbr_samples, t_func)

@time ens_sv_res = get_ensemble_average_sv_res_qn(; nbr_dots_res_list, ham_param_funcs, nbr_samples, t_func)

sv_res = fig = Figure(size = (1500, 600))
plot_smallest_sv_qn!(fig[1, 1], ens_sv_res)
plot_mean_sv_qn!(fig[1, 2], ens_sv_res)
plot_condition_number_qn!(fig[1, 3], ens_sv_res)
fig
