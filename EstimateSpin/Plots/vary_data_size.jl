function mse_against_training_size(
        nbr_train_states_list, mse, mse_diff, weight_diff, S; xscale = log10, title = "", legend = true)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "Number of training states",
        ylabel = "MSE", yscale = log10, xscale = xscale, title)
    scatter!(ax, nbr_train_states_list, mse, label = "MSE")
    lines!(ax, nbr_train_states_list, mse)
    scatter!(ax, nbr_train_states_list, weight_diff,
        label = L"\frac{norm(W_X - W_X^{theory})}{norm(W_X^{theory})}")
    lines!(ax, nbr_train_states_list, weight_diff)
    scatter!(ax, nbr_train_states_list, mse_diff,
        label = L"\frac{norm(MSE - MSE_{theory})}{norm(MSE_{theory})}")
    lines!(ax, nbr_train_states_list, mse_diff)
    vlines!(ax, [16], color = :grey, linestyle = :dash, label = "16 training states")
    vlines!(ax, [size(S, 1)], color = :black, linestyle = :dash,
        label = "$(size(S, 1)) training states = rows in S")
    legend && axislegend(ax)
    return fig
end

function mse_against_test_size(
        nbr_test_states_list, mse, mse_diff, xscale = log10)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Number of test states",
        ylabel = "MSE", yscale = log10, xscale = xscale)
    scatter!(ax, nbr_test_states_list, mse, label = "MSE")
    lines!(ax, nbr_test_states_list, mse)
    scatter!(ax, nbr_test_states_list, mse_diff,
        label = L"\frac{norm(MSE - MSE_{theory})}{norm(MSE_{theory})}")
    lines!(ax, nbr_test_states_list, mse_diff)
    axislegend(ax)
    return fig
end

"Least-squares fit of log10(err) = a + k*log10(N) over N ≥ N_min"
function loglog_fit(N, err, N_min)
    idx = findall(≥(N_min), N)
    X = [ones(length(idx)) log10.(N[idx])]
    a, k = X \ log10.(err[idx])
    return a, k
end

"Relative weight error against training size and relative MSE error against test size.
If `N_fit_min` is given, both curves get a log-log fit over N ≥ N_fit_min."
function mse_W_convergence(
        nbr_train_states_list, nbr_test_states_list, weight_diff, mse_diff, S;
        xscale = log10, title = "", N_fit_min = nothing)
    fig = Figure(size = (500, 280))
    ax = Axis(
        fig[1, 1]; xlabel = L"Number of states $N$", ylabel = "Relative error to theory",
        yscale = log10, xscale, title, xgridvisible = false, ygridvisible = false)
    c_w, c_mse = Makie.wong_colors()[1], Makie.wong_colors()[2]
    vlines!(ax, [size(S, 1)], color = (:black, 0.6), linestyle = :dash)
    scatterlines!(ax, nbr_train_states_list, weight_diff, color = c_w, markersize = 5,
        label = L"$\frac{\Vert W_X - W_X^\mathrm{theory} \Vert}{\Vert W_X^\mathrm{theory} \Vert}$")
    scatterlines!(ax, nbr_test_states_list, mse_diff, color = c_mse, markersize = 5,
        label = L"$ \frac{\Vert \mathrm{MSE} - \mathrm{MSE}_\mathrm{theory} \Vert}{\Vert \mathrm{MSE}_\mathrm{theory} \Vert}$")
    if !isnothing(N_fit_min)
        for (N, err, c) in ((nbr_train_states_list, weight_diff, c_w),
            (nbr_test_states_list, mse_diff, c_mse))
            a, k = loglog_fit(N, err, N_fit_min)
            fit(n) = exp10(a) * n^k
            N_line = exp10.(range(log10(N_fit_min), log10(maximum(N)), length = 100))
            lines!(ax, N_line, fit.(N_line), color = c, linestyle = :dash, linewidth = 1.5)
            # Slope label below the middle of the fit line
            N_mid = sqrt(N_fit_min * maximum(N))
            text!(ax, N_mid, fit(N_mid); text = L"\propto N^{%$(round(k, digits = 2))}",
                color = c, align = (:right, :top), offset = (0, -8))
        end
    end
    Legend(fig[1, 2], ax, framevisible = false)
    return fig
end
