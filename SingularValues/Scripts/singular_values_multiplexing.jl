using QDELM, LinearAlgebra, CairoMakie, Statistics
includet("..\\Core\\singular_values_multiplexing.jl")
includet("..\\Plots\\singular_values_multiplexing.jl")

## ============ System =====================
nbr_dots_res = 4
qn_res = 2
sys = tight_binding_system(2, nbr_dots_res, qn_res)
measurements = QDELM.charge_probabilities_01(sys)
Pm, Pm_dict = QDELM.pauli_matrix(sys.Hs_main, sys.H_main)
B = 1/2 .* Pm[:, 2:end]

function custom_param_functions()
    ϵ_main = 0.5
    ϵ_res = 1
    ϵb = [0, 0, 1]
    u_intra = 1.0
    t = 1.0
    t_so = 10
    u_inter = 1.0
    QDELM.ParamFunctions(
        ϵ_func_main = () -> ϵ_main,
        ϵ_func_res = () -> rand() * ϵ_res,
        ϵb_func = () -> ϵb,
        u_intra_func = () -> u_intra * 10 + rand(),
        t_func = () -> rand() * t,
        t_so_func = () -> t_so * rand(),
        u_inter_func = () -> u_inter * rand()
    )
end

## ================ Multiplexing =====================
M_max = 300
S_list_t = time_multiplexing(;
    sys, measurements, M_max, seed = 7284, param_funcs = custom_param_functions)
S_list_h = hamiltonian_multiplexing(;
    sys, measurements, M_max, seed = 8310, param_funcs = custom_param_functions)

sv_t = hcat(svd_lists_multiplexing(S_list_t, B)...)
sv_h = hcat(svd_lists_multiplexing(S_list_h, B)...)

sv_t_sqrt = rescale_by_m(sv_t)
sv_h_sqrt = rescale_by_m(sv_h)

κ_t = [κ(sv_t[1, m], sv_t[end, m]) for m in 1:M_max]
κ_h = [κ(sv_h[1, m], sv_h[end, m]) for m in 1:M_max]

mean_sv_t = [mean(sv_t[:, m]) for m in 1:M_max]
mean_sv_h = [mean(sv_h[:, m]) for m in 1:M_max]

mean_sv_t_sqrt = rescale_by_m(reshape(mean_sv_t, 1, :))
mean_sv_h_sqrt = rescale_by_m(reshape(mean_sv_h, 1, :))

## ==== Plot singular values ======= 
fig = Figure(size = (900, 600))
plot_sv_m!(fig[1, 1], sv_t, mean_sv_t, M_max; title = "Time multiplexing")
plot_sv_m!(fig[1, 2], sv_h, mean_sv_h, M_max; title = "Hamiltonian multiplexing")
plot_κ_m!(fig[2, 1], κ_t, M_max; title = "Time multiplexing")
plot_κ_m!(fig[2, 2], κ_h, M_max; title = "Hamiltonian multiplexing")
fig

fig = Figure(size = (900, 600))
plot_sv_m!(fig[1, 1], sv_t_sqrt, vec(mean_sv_t_sqrt), M_max; title = "Time multiplexing")
plot_sv_m!(
    fig[1, 2], sv_h_sqrt, vec(mean_sv_h_sqrt), M_max; title = "Hamiltonian multiplexing")
fig

fig = Figure(size = (900, 600))
plot_sv_lines!(fig[1, 1], sv_t_sqrt[:, end], title = "Time multiplexing")
plot_sv_lines!(fig[1, 2], sv_h_sqrt[:, end], title = "Hamiltonian multiplexing")
fig
save("Figures\\singular_values_multiplexing_rescaled_lines_tso=10.png", fig)
##
S_t_maxM = vcat(S_list_t...)
S_h_maxM = vcat(S_list_h...)

F_t = svd(S_t_maxM * B)
F_h = svd(S_h_maxM * B)

fig = Figure(size = (900, 600))
plot_pauli_overlaps!(fig[1, 1], F_h, title = "Hamiltonian multiplexing")
fig

fig = Figure(size = (900, 600))
plot_pauli_overlaps!(fig[1, 1], F_t, title = "Time multiplexing")
fig
