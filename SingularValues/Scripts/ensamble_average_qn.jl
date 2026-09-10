includet("..\\Core\\singular_values_multiplexing.jl")
includet("..\\Plots\\singular_values_multiplexing.jl")
using CairoMakie

ham_param_funcs = nbr_samples = 100
t_func = () -> 100
nbr_dots_res_list = [1, 2, 3, 4]

ens_sv_res_default = get_ensemble_average_sv_res_qn(;
    nbr_dots_res_list, ham_param_funcs = QDELM.random_param_functions(), nbr_samples, t_func)

ens_sv_res_t100 = get_ensemble_average_sv_res_qn(;
    nbr_dots_res_list, ham_param_funcs = QDELM.random_param_functions(t = 100), nbr_samples, t_func)

ens_sv_res_tso001 = get_ensemble_average_sv_res_qn(;
    nbr_dots_res_list, ham_param_funcs = QDELM.random_param_functions(t_so = 0.01), nbr_samples, t_func)

ens_sv_res_time1000 = get_ensemble_average_sv_res_qn(;
    nbr_dots_res_list, ham_param_funcs = QDELM.random_param_functions(), nbr_samples, t_func = () -> 1000)

ens_sv_res_unitra0_uinter0 = get_ensemble_average_sv_res_qn(;
    nbr_dots_res_list, ham_param_funcs = QDELM.random_param_functions(u_intra = 0.0, u_inter = 0.0),
    nbr_samples, t_func)

fig = Figure(size = (1500, 400*5))
plot_sv_stats_qn(fig[1, 1:4], ens_sv_res; title = "Default parameters")
plot_sv_stats_qn(fig[2, 1:4], ens_sv_res_t100; title = "t = 100")
plot_sv_stats_qn(fig[3, 1:4], ens_sv_res_tso001; title = "t_so = 0.01")
plot_sv_stats_qn(fig[4, 1:4], ens_sv_res_time1000; title = "time = 1000")
plot_sv_stats_qn(fig[5, 1:4], ens_sv_res_unitra0_uinter0; title = "u_intra = 0.0, u_inter = 0.0")
fig