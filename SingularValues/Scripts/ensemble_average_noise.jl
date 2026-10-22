## Load data
includet("..\\Plots\\ensemble_average_params.jl")

sv_list_setting_so_naive = load(
    "SingularValues\\Data\\sv_list_setting_res3_naive_so.jld2", "sv_list_setting_so")
sv_list_setting_so_iso = load(
    "SingularValues\\Data\\sv_list_setting_res3_iso_so.jld2", "sv_list_setting_so")
sv_list_setting_so_shot = load(
    "SingularValues\\Data\\sv_list_setting_res3_shot_so.jld2", "sv_list_setting_so")
parameter_range = load(
    "SingularValues\\Data\\sv_list_setting_res3_iso_so.jld2", "parameter_range")

figssv = Figure(size = (900, 600))
Label(figssv[0, 1:3], "Smallest Singular Value of Ensemble Average", fontsize = 20)
plot_sv_against_param_range!(figssv[1, 1], sv_list_setting_so_naive, parameter_range,
    "Varying SO"; axisledgend = false, xscale = identity, sv_func = minimum)
plot_sv_against_param_range!(figssv[1, 2], sv_list_setting_so_iso,
    parameter_range, "Varying SO (ISO)"; axisledgend = false, xscale = identity, sv_func = minimum)
plot_sv_against_param_range!(figssv[1, 3], sv_list_setting_so_shot,
    parameter_range, "Varying SO (SHOT)"; axisledgend = false, xscale = identity, sv_func = minimum)

figssv