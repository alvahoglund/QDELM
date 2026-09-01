function mse_against_training_size(
        nbr_train_states_list, mse, mse_diff, weight_diff, S, xscale = log10)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Number of training states",
        ylabel = "MSE", yscale = log10, xscale = xscale)
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
    axislegend(ax)
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

