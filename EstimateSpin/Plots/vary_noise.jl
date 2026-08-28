function mse_against_noise(mse_list, mse_theory_list, target_labels, sv_overlaps, b)
    fig = Figure(size = (800, 300))
    for (i, target) in enumerate(target_labels)
        ax = Axis(fig[1, i], xlabel = "Noise level (σE)", ylabel = "MSE",
            title = "Estimating $(target[1]) ⊗ $(target[2])", xscale = log10,
            xgridvisible = false, ygridvisible = false)
        lines!(ax, σE_list, getindex.(mse_theory_list, i), label = "Predicted MSE")
        lines!(ax, σE_list, getindex.(mse_list, i), label = "MSE")
        vlines!(ax, sv_overlaps.vals .* sqrt(b),
            color = vec(abs2.(sv_overlaps.overlaps[i, :])), colormap = :blues,
            colorrange = (0, maximum(abs2.(sv_overlaps.overlaps))),
            linestyle = :dash, label = "√b·σS")
        axislegend(ax, position = :lt)
    end
    return fig
end