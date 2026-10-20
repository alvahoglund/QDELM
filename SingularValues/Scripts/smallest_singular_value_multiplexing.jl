using QDELM, LinearAlgebra, CairoMakie, Statistics
includet("..\\Core\\singular_values_multiplexing.jl")

## ============ System =====================
nbr_dots_main = 2
nbr_dots_res = 6
qn_res = 1
b = 0.0147
M_max = 500

sys = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_res)
measurements = ChargeMeasurements01(sys, 1)
Pm, _ = QDELM.pauli_matrix(sys.Hs_main, sys.H_main)
B = 1/2 .* Pm[:, 2:end]
noise = ShotNoise(1.0, measurements)
param_funcs = QDELM.random_param_functions(t = 100)

## ================ Multiplexing =====================
σ_min_time = minimum.(svd_multiplexing(time_multiplexing,
    sys, measurements, B, b, noise, param_funcs = param_funcs))
σ_min_hamiltonian = minimum.(svd_multiplexing(hamiltonian_multiplexing,
    sys, measurements, B, b, noise, param_funcs = param_funcs))
σ_min_identical = minimum.(svd_multiplexing(identical_multiplexing,
    sys, measurements, B, b, noise; m0 = 1, param_funcs = param_funcs))

## ================ Linear fit in log-log =====================
# Fit log10(σ_min) = a + k*log10(m) over the asymptotic region m ≥ m_fit_min
m_fit_min = 100
function loglog_fit(σ_min, m_fit_min)
    m_fit = m_fit_min:length(σ_min)
    X = [ones(length(m_fit)) log10.(m_fit)]
    a, k = X \ log10.(σ_min[m_fit])
    return a, k
end

# Smallest singular value of the ensemble-averaged G = sum(G)/M, i.e. σ_min(M) / √M
ens_avg(σ_min) = σ_min[end] / sqrt(length(σ_min))

mp_methods = [("Identical", σ_min_identical), ("Time", σ_min_time),
    ("Hamiltonian", σ_min_hamiltonian)]
fits = [loglog_fit(σ_min, m_fit_min) for (_, σ_min) in mp_methods]

println("res = $nbr_dots_res dots, qn = $qn_res:")
for ((name, σ_min), (a, k)) in zip(mp_methods, fits)
    println("  $name multiplexing: slope = $(round(k, digits = 3)), ",
        "ensemble average σ_min = $(round(ens_avg(σ_min), sigdigits = 3))")
end

## ================ Plot =====================
colors = Makie.wong_colors()
fig = Figure(size = (600, 300))
ax = Axis(fig[1, 1], xlabel = "Multiplexing Index", ylabel = "Smallest Singular Value",
    xscale = log10, yscale = log10)
ms = 1:M_max
m_line = range(1, M_max, length = 200)
marker_sizes = 10

for (i, ((name, σ_min), (a, k))) in enumerate(zip(mp_methods, fits))
    scatter!(ax, ms, σ_min, markersize = marker_sizes, color = colors[i], label = name)
    lines!(ax, ms, σ_min)
    lines!(ax, m_line, 10^a .* m_line .^ k, linestyle = :dash, color = colors[i])
end
axislegend(ax, position = :rb)
fig
save("Figures\\smallest_singular_value_multiplexing.png", fig)