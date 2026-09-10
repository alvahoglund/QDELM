includet("..\\Plots\\ensemble_average_params.jl")

#LOAD DATA
sv_list_setting_so = load("SingularValues\\Data\\sv_list_setting_res2_so.jld2", "sv_list_setting_so")
sv_list_setting_t = load("SingularValues\\Data\\sv_list_setting_res2_t.jld2", "sv_list_setting_t")
sv_list_setting_u_intra = load(
    "SingularValues\\Data\\sv_list_setting_res2_u_intra.jld2", "sv_list_setting_u_intra")
sv_list_setting_u_inter = load(
    "SingularValues\\Data\\sv_list_setting_res2_u_inter.jld2", "sv_list_setting_u_inter")
sv_list_setting_ϵ = load("SingularValues\\Data\\sv_list_setting_res2_ϵ.jld2", "sv_list_setting_ϵ")
sv_list_setting_ϵb = load("SingularValues\\Data\\sv_list_setting_res2_ϵb.jld2", "sv_list_setting_ϵb")
parameter_range = load("SingularValues\\Data\\sv_list_setting_res2_so.jld2", "parameter_range")

## PLOT
figssv = Figure(size = (900, 600))
Label(figssv[0, 1:2], "Smallest Singular Value of Ensemble Average", fontsize = 20)
plot_sv_against_param_range!(figssv[1, 1], sv_list_setting_so, parameter_range,
    "Varying SO"; axisledgend = false, xscale = identity, sv_func = minimum)
plot_sv_against_param_range!(figssv[1, 2], sv_list_setting_t,
    parameter_range, "Varying t"; axisledgend = false, xscale = identity, sv_func = minimum)
plot_sv_against_param_range!(figssv[2, 1], sv_list_setting_u_intra, parameter_range,
    "Varying u_intra"; axisledgend = false, xscale = identity, sv_func = minimum)
plot_sv_against_param_range!(figssv[2, 2], sv_list_setting_u_inter, parameter_range,
    "Varying u_inter"; axisledgend = false, xscale = identity, sv_func = minimum)
plot_sv_against_param_range!(figssv[3, 1], sv_list_setting_ϵ, parameter_range,
    "Varying ϵ"; axisledgend = false, xscale = identity, sv_func = minimum)
plot_sv_against_param_range!(figssv[3, 2], sv_list_setting_ϵb, parameter_range,
    "Varying ϵb"; axisledgend = false, xscale = identity, sv_func = minimum)
add_legend_settings!(figssv[1, 3], sort(collect(keys(sv_list_setting_ϵb))))
figssv

figmean = Figure(size = (900, 600))
Label(figmean[0, 1:2], "Mean Singular Value of Ensemble Average", fontsize = 20)
plot_sv_against_param_range!(figmean[1, 1], sv_list_setting_so, parameter_range,
    "Varying SO"; axisledgend = false, xscale = identity, sv_func = mean)
plot_sv_against_param_range!(figmean[1, 2], sv_list_setting_t,
    parameter_range, "Varying t"; axisledgend = false, xscale = identity, sv_func = mean)
plot_sv_against_param_range!(figmean[2, 1], sv_list_setting_u_intra, parameter_range,
    "Varying u_intra"; axisledgend = false, xscale = identity, sv_func = mean)
plot_sv_against_param_range!(figmean[2, 2], sv_list_setting_u_inter, parameter_range,
    "Varying u_inter"; axisledgend = false, xscale = identity, sv_func = mean)
plot_sv_against_param_range!(figmean[3, 1], sv_list_setting_ϵ, parameter_range,
    "Varying ϵ"; axisledgend = false, xscale = identity, sv_func = mean)
plot_sv_against_param_range!(figmean[3, 2], sv_list_setting_ϵb, parameter_range,
    "Varying ϵb"; axisledgend = false, xscale = identity, sv_func = mean)
add_legend_settings!(figmean[1, 3], sort(collect(keys(sv_list_setting_ϵb))))
figmean

figc = Figure(size = (900, 600))
Label(figc[0, 1:2], "Condition number of SB", fontsize = 20)
yticks = Makie.SymlogTicks()
plot_sv_against_param_range!(figc[1, 1], sv_list_setting_so, parameter_range,
    "Varying SO"; axisledgend = false, xscale = identity,
    sv_func = condition_number, symlog_exp = 0, yticks = yticks)
plot_sv_against_param_range!(figc[1, 2], sv_list_setting_t,
    parameter_range, "Varying t"; axisledgend = false, xscale = identity,
    sv_func = condition_number, symlog_exp = 0, yticks = yticks)
plot_sv_against_param_range!(
    figc[2, 1], sv_list_setting_u_intra, parameter_range,
    "Varying u_intra"; axisledgend = false, xscale = identity,
    sv_func = condition_number, symlog_exp = 0, yticks = yticks)
plot_sv_against_param_range!(
    figc[2, 2], sv_list_setting_u_inter, parameter_range,
    "Varying u_inter"; axisledgend = false, xscale = identity,
    sv_func = condition_number, symlog_exp = 0, yticks = yticks)
plot_sv_against_param_range!(figc[3, 1], sv_list_setting_ϵ, parameter_range,
    "Varying ϵ"; axisledgend = false, xscale = identity,
    sv_func = condition_number, symlog_exp = 0, yticks = yticks)
plot_sv_against_param_range!(figc[3, 2], sv_list_setting_ϵb, parameter_range,
    "Varying ϵb"; axisledgend = false, xscale = identity,
    sv_func = condition_number, symlog_exp = 0, yticks = yticks)
add_legend_settings!(figc[1, 3], sort(collect(keys(sv_list_setting_ϵb))))
figc

figssv
figmean
figc

## Plot t and tso

sv_list_setting_t = load("SingularValues\\Data\\sv_list_setting_t.jld2", "sv_list_setting_t")
sv_list_setting_so = load("SingularValues\\Data\\sv_list_setting_so.jld2", "sv_list_setting_so")

settings_plot = [(2, 0), (2, 1), (2, 2), (6, 0), (6, 1), (6, 2)]

sv_list_setting_so_plot = Dict{Tuple{Int, Int}, Vector{Vector{Float64}}}()
sv_list_setting_t_plot = Dict{Tuple{Int, Int}, Vector{Vector{Float64}}}()
for setting in settings_plot
    sv_list_setting_so_plot[setting] = sv_list_setting_so[setting]
    sv_list_setting_t_plot[setting] = sv_list_setting_t[setting]
end

figssv = Figure(size = (900, 600))
Label(figssv[0, 1:2], "Smallest Singular Value of Ensemble Average", fontsize = 20)
plot_sv_against_param_range!(figssv[1, 1], sv_list_setting_so_plot, parameter_range,
    "Varying SO"; axisledgend = false, xscale = identity, sv_func = minimum)
plot_sv_against_param_range!(figssv[1, 2], sv_list_setting_t_plot,
    parameter_range, "Varying t"; axisledgend = false, xscale = identity, sv_func = minimum)
add_legend_settings!(figssv[1, 3], sort(collect(keys(sv_list_setting_so_plot))))

figmean = Figure(size = (900, 600))
Label(figmean[0, 1:2], "Mean Singular Value of Ensemble Average", fontsize = 20)
plot_sv_against_param_range!(figmean[1, 1], sv_list_setting_so_plot, parameter_range,
    "Varying SO"; axisledgend = false, xscale = identity, sv_func = mean)
plot_sv_against_param_range!(figmean[1, 2], sv_list_setting_t_plot,
    parameter_range, "Varying t"; axisledgend = false, xscale = identity, sv_func = mean)
add_legend_settings!(figmean[1, 3], sort(collect(keys(sv_list_setting_so_plot))))

figc = Figure(size = (900, 600))
Label(figc[0, 1:2], "Condition Number of Ensemble Average", fontsize = 20)
yticks = Makie.SymlogTicks()
plot_sv_against_param_range!(figc[1, 1], sv_list_setting_so_plot, parameter_range,
    "Varying SO"; axisledgend = false, xscale = identity,
    sv_func = condition_number, symlog_exp = 0, yticks = yticks, ylims = (nothing, 10^7))
plot_sv_against_param_range!(figc[1, 2], sv_list_setting_t_plot,
    parameter_range, "Varying t"; axisledgend = false, xscale = identity,
    sv_func = condition_number, symlog_exp = 0, yticks = yticks, ylims = (nothing, 10^7))
add_legend_settings!(figc[1, 3], sort(collect(keys(sv_list_setting_so_plot))))

figmedian = Figure(size = (900, 600))
Label(figmedian[0, 1:2], "Median Singular Value of Ensemble Average", fontsize = 20)
plot_sv_against_param_range!(figmedian[1, 1], sv_list_setting_so_plot, parameter_range,
    "Varying SO"; axisledgend = false, xscale = identity, sv_func = median)
plot_sv_against_param_range!(figmedian[1, 2], sv_list_setting_t_plot,
    parameter_range, "Varying t"; axisledgend = false, xscale = identity, sv_func = median)
add_legend_settings!(figmedian[1, 3], sort(collect(keys(sv_list_setting_so_plot))))

figssv
figmean
figc
figmedian

## Plot all params for one setting in one plot

sv_list_setting_so = load("SingularValues\\Data\\sv_list_setting_res3_so.jld2", "sv_list_setting_so")
sv_list_setting_t = load("SingularValues\\Data\\sv_list_setting_res3_t.jld2", "sv_list_setting_t")
sv_list_setting_u_intra = load(
    "SingularValues\\Data\\sv_list_setting_res3_u_intra.jld2", "sv_list_setting_u_intra")
sv_list_setting_u_inter = load(
    "SingularValues\\Data\\sv_list_setting_res3_u_inter.jld2", "sv_list_setting_u_inter")
sv_list_setting_ϵ = load("SingularValues\\Data\\sv_list_setting_res3_ϵ.jld2", "sv_list_setting_ϵ")
sv_list_setting_ϵb = load("SingularValues\\Data\\sv_list_setting_res3_ϵb.jld2", "sv_list_setting_ϵb")
parameter_range = load("SingularValues\\Data\\sv_list_setting_res3_so.jld2", "parameter_range")

setting = (3, 2)

sv_lists_params = Dict{String, Vector{Vector{Float64}}}()
sv_lists_params["SO"] = sv_list_setting_so[setting]
sv_lists_params["t"] = sv_list_setting_t[setting]
sv_lists_params["u_intra"] = sv_list_setting_u_intra[setting]
sv_lists_params["u_inter"] = sv_list_setting_u_inter[setting]
sv_lists_params["ϵ"] = sv_list_setting_ϵ[setting]
sv_lists_params["ϵb"] = sv_list_setting_ϵb[setting]

fig = Figure(size = (900, 600))
Label(fig[0, 1],
    "Smallest Singular Value of Ensemble Average for res = $(setting[1]), qn = $(setting[2])",
    fontsize = 20)
plot_sv_against_param_range!(
    fig[1, 1], sv_lists_params, parameter_range, "Varying Hamiltonian parameters";
    axisledgend = true, xscale = identity, sv_func = median, legendposistion = :rb)
fig