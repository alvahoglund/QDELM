includet("..\\Plots\\ensemble_average_params.jl")

## PLOT 
sv_list_setting_so = load("SingularValues\\Data\\sv_list_setting_res2_so.jld2", "sv_list_setting_so")
sv_list_setting_t = load("SingularValues\\Data\\sv_list_setting_res2_t.jld2", "sv_list_setting_t")
sv_list_setting_u_intra = load(
    "SingularValues\\Data\\sv_list_setting_res2_u_intra.jld2", "sv_list_setting_u_intra")
sv_list_setting_u_inter = load(
    "SingularValues\\Data\\sv_list_setting_res2_u_inter.jld2", "sv_list_setting_u_inter")
sv_list_setting_ϵ = load("SingularValues\\Data\\sv_list_setting_res2_ϵ.jld2", "sv_list_setting_ϵ")
sv_list_setting_ϵb = load("SingularValues\\Data\\sv_list_setting_res2_ϵb.jld2", "sv_list_setting_ϵb")
parameter_range = load("SingularValues\\Data\\sv_list_setting_res2_so.jld2", "parameter_range")

fig = Figure(size = (900, 600))
Label(fig[0, 1:2], "Smallest Singular Value of Ensemble Average", fontsize = 20)
plot_sv_against_param_range!(fig[1, 1], sv_list_setting_so, parameter_range,
    "Varying SO"; axisledgend = false, xscale = identity, sv_func = minimum)
plot_sv_against_param_range!(fig[1, 2], sv_list_setting_t,
    parameter_range, "Varying t"; axisledgend = false, xscale = identity, sv_func = minimum)
plot_sv_against_param_range!(fig[2, 1], sv_list_setting_u_intra, parameter_range,
    "Varying u_intra"; axisledgend = false, xscale = identity, sv_func = minimum)
plot_sv_against_param_range!(fig[2, 2], sv_list_setting_u_inter, parameter_range,
    "Varying u_inter"; axisledgend = false, xscale = identity, sv_func = minimum)
plot_sv_against_param_range!(fig[3, 1], sv_list_setting_ϵ, parameter_range,
    "Varying ϵ"; axisledgend = false, xscale = identity, sv_func = minimum)
plot_sv_against_param_range!(fig[3, 2], sv_list_setting_ϵb, parameter_range,
    "Varying ϵb"; axisledgend = false, xscale = identity, sv_func = minimum)
add_legend_settings!(fig[1, 3], sort(collect(keys(sv_list_setting_ϵb))))
save("Figures\\ensemble_average_params_sv_min_res2.png", fig)

fig = Figure(size = (900, 600))
Label(fig[0, 1:2], "Mean Singular Value of Ensemble Average", fontsize = 20)
plot_sv_against_param_range!(fig[1, 1], sv_list_setting_so, parameter_range,
    "Varying SO"; axisledgend = false, xscale = identity, sv_func = mean)
plot_sv_against_param_range!(fig[1, 2], sv_list_setting_t,
    parameter_range, "Varying t"; axisledgend = false, xscale = identity, sv_func = mean)
plot_sv_against_param_range!(fig[2, 1], sv_list_setting_u_intra, parameter_range,
    "Varying u_intra"; axisledgend = false, xscale = identity, sv_func = mean)
plot_sv_against_param_range!(fig[2, 2], sv_list_setting_u_inter, parameter_range,
    "Varying u_inter"; axisledgend = false, xscale = identity, sv_func = mean)
plot_sv_against_param_range!(fig[3, 1], sv_list_setting_ϵ, parameter_range,
    "Varying ϵ"; axisledgend = false, xscale = identity, sv_func = mean)
plot_sv_against_param_range!(fig[3, 2], sv_list_setting_ϵb, parameter_range,
    "Varying ϵb"; axisledgend = false, xscale = identity, sv_func = mean)
add_legend_settings!(fig[1, 3], sort(collect(keys(sv_list_setting_ϵb))))
save("Figures\\ensemble_average_params_sv_mean_res2.png", fig)

fig = Figure(size = (900, 600))
Label(fig[0, 1:2], "Condition number of SB", fontsize = 20)
yticks = Makie.SymlogTicks()
plot_sv_against_param_range!(fig[1, 1], sv_list_setting_so, parameter_range,
    "Varying SO"; axisledgend = false, xscale = identity,
    sv_func = condition_number, symlog_exp = 0, yticks = yticks)
plot_sv_against_param_range!(fig[1, 2], sv_list_setting_t,
    parameter_range, "Varying t"; axisledgend = false, xscale = identity,
    sv_func = condition_number, symlog_exp = 0, yticks = yticks)
plot_sv_against_param_range!(
    fig[2, 1], sv_list_setting_u_intra, parameter_range,
    "Varying u_intra"; axisledgend = false, xscale = identity,
    sv_func = condition_number, symlog_exp = 0, yticks = yticks)
plot_sv_against_param_range!(
    fig[2, 2], sv_list_setting_u_inter, parameter_range,
    "Varying u_inter"; axisledgend = false, xscale = identity,
    sv_func = condition_number, symlog_exp = 0, yticks = yticks)
plot_sv_against_param_range!(fig[3, 1], sv_list_setting_ϵ, parameter_range,
    "Varying ϵ"; axisledgend = false, xscale = identity,
    sv_func = condition_number, symlog_exp = 0, yticks = yticks)
plot_sv_against_param_range!(fig[3, 2], sv_list_setting_ϵb, parameter_range,
    "Varying ϵb"; axisledgend = false, xscale = identity,
    sv_func = condition_number, symlog_exp = 0, yticks = yticks)
add_legend_settings!(fig[1, 3], sort(collect(keys(sv_list_setting_ϵb))))
save("Figures\\ensemble_average_params_sv_condition_number_res2.png", fig)
