includet("..\\Core\\singular_values_multiplexing.jl")
using CairoMakie, Statistics, JLD2

nbr_samples = 100
#parameter_range = exp10.(range(log10(0.001), log10(20), length = 50))
parameter_range = range(0, 15, length = 50)
t_func = () -> 100
settings = [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6)]

# Vary SO
ham_param_func_list_so = [QDELM.random_param_functions(t_so = p)
                          for p in parameter_range]
sv_list_setting_so = get_ensemble_average_sv_params_settings(;
    settings, ham_param_func_list = ham_param_func_list_so, nbr_samples, t_func)

save("SingularValues\\Data\\sv_list_setting_res3_so.jld2",
    "sv_list_setting_so", sv_list_setting_so,
    "parameter_range", parameter_range, "settings", settings, "nbr_samples", nbr_samples, "t_func", t_func)

# Vary t
ham_param_func_list_t = [QDELM.random_param_functions(t = p)
                         for p in parameter_range]
sv_list_setting_t = get_ensemble_average_sv_params_settings(;
    settings, ham_param_func_list = ham_param_func_list_t, nbr_samples, t_func)
save("SingularValues\\Data\\sv_list_setting_res3_t.jld2",
    "sv_list_setting_t", sv_list_setting_t,
    "parameter_range", parameter_range, "settings", settings, "nbr_samples", nbr_samples, "t_func", t_func)

# Vary u_intra
ham_param_func_list_u_intra = [QDELM.random_param_functions(u_intra = p)
                               for p in parameter_range]
sv_list_setting_u_intra = get_ensemble_average_sv_params_settings(;
    settings, ham_param_func_list = ham_param_func_list_u_intra, nbr_samples, t_func)
save("SingularValues\\Data\\sv_list_setting_res3_u_intra.jld2",
    "sv_list_setting_u_intra", sv_list_setting_u_intra,
    "parameter_range", parameter_range, "settings", settings, "nbr_samples", nbr_samples, "t_func", t_func)

# Vary u_inter
ham_param_func_list_u_inter = [QDELM.random_param_functions(u_inter = p)
                               for p in parameter_range]
sv_list_setting_u_inter = get_ensemble_average_sv_params_settings(;
    settings, ham_param_func_list = ham_param_func_list_u_inter, nbr_samples, t_func)
save("SingularValues\\Data\\sv_list_setting_res3_u_inter.jld2",
    "sv_list_setting_u_inter", sv_list_setting_u_inter,
    "parameter_range", parameter_range, "settings", settings, "nbr_samples", nbr_samples, "t_func", t_func)

#Vary ϵ
ham_param_func_list_ϵ = [QDELM.random_param_functions(ϵ_main = p, ϵ_res = p)
                         for p in parameter_range]
sv_list_setting_ϵ = get_ensemble_average_sv_params_settings(;
    settings, ham_param_func_list = ham_param_func_list_ϵ, nbr_samples, t_func)
save("SingularValues\\Data\\sv_list_setting_res3_ϵ.jld2",
    "sv_list_setting_ϵ", sv_list_setting_ϵ,
    "parameter_range", parameter_range, "settings", settings, "nbr_samples", nbr_samples, "t_func", t_func)

#Vary ϵb
ham_param_func_list_ϵb = [QDELM.random_param_functions(ϵb = [0, 0, p])
                          for p in parameter_range]
sv_list_setting_ϵb = get_ensemble_average_sv_params_settings(;
    settings, ham_param_func_list = ham_param_func_list_ϵb, nbr_samples, t_func)
save("SingularValues\\Data\\sv_list_setting_res3_ϵb.jld2",
    "sv_list_setting_ϵb", sv_list_setting_ϵb,
    "parameter_range", parameter_range, "settings", settings, "nbr_samples", nbr_samples, "t_func", t_func)
