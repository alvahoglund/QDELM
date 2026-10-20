includet("..\\Core\\singular_values_multiplexing.jl")
includet("..\\Plots\\singular_values_multiplexing.jl")
using CairoMakie, Random

## ============ Vary params ====================
ham_param_funcs = QDELM.random_param_functions()
nbr_samples = 500
t_func = () -> 100
nbr_dots_res_list = [1, 2]
noise_model = QDELM.ShotNoise
b = 0.0147   # Hilbert–Schmidt ensemble parameter, only used by ShotNoise
ens_sv_res_default = get_ensemble_average_sv_res_qn(;
    nbr_dots_res_list, ham_param_funcs = QDELM.random_param_functions(), noise_model, b, nbr_samples, t_func)
fig = Figure(size = (1500, 400))
plot_sv_stats_qn(fig[1, 1:4], ens_sv_res_default; title = "Default parameters")
fig

## ============ Vary measurement set ====================
ham_param_funcs = QDELM.random_param_functions()
nbr_samples = 200
t_func = () -> 100
nbr_dots_res_list = [1, 2]
b = 0.0147   # Hilbert–Schmidt ensemble parameter, only used by ShotNoise
noise_model = QDELM.ShotNoise
msfs = [(QDELM.ChargeMeasurements01, "ChargeMeasurements01"),
    (QDELM.ChargeMeasurements12, "ChargeMeasurements12"),
    (QDELM.ChargeMeasurements012, "ChargeMeasurements012")]

ens_sv_ms = map(msfs) do (msf, _)
    Random.seed!(1234)
    get_ensemble_average_sv_res_qn(;
        nbr_dots_res_list, ham_param_funcs, nbr_samples, t_func,
        noise_model, b, msf)
end

fig = Figure(size = (700, 300 * length(ens_sv_ms)))
for (i, ((_, title), ens_sv)) in enumerate(zip(msfs, ens_sv_ms))
    gl = fig[i, 1:2]
    Label(gl[0, 1:2], title, fontsize = 20)
    plot_sv_qn!(gl[1, 1], ens_sv, minimum;
        title = "Minimum singular value of ensemble average",
        ylabel = "Minimum singular value")
    plot_sv_qn!(gl[1, 2], ens_sv, mean;
        title = "Mean singular value of ensemble average",
        ylabel = "Mean singular value")
end
fig

## ============= Vary noise model =====================

ham_param_funcs = QDELM.random_param_functions()
nbr_samples = 100
t_func = () -> 100
nbr_dots_res_list = [1, 2]
b = 0.0147   # Hilbert–Schmidt ensemble parameter, only used by ShotNoise
msf = QDELM.ChargeMeasurements12
noise_models = [
    (QDELM.NaiveNoise, "Naive noise (unwhitened)"),
    (QDELM.IsotropicNoise, "Isotropic noise"),
    (QDELM.ShotNoise, "Shot noise")]

# Same seed for every model, so all models see the same sampled Hamiltonians
ens_sv_noise = map(noise_models) do (noise_model, _)
    Random.seed!(1234)
    get_ensemble_average_sv_res_qn(;
        nbr_dots_res_list, ham_param_funcs, nbr_samples, t_func,
        noise_model, b, msf = msf)
end

fig = Figure(size = (1500, 400 * length(noise_models)))
for (i, ((_, title), ens_sv)) in enumerate(zip(noise_models, ens_sv_noise))
    plot_sv_stats_qn(fig[i, 1:4], ens_sv; title)
end
fig
