function mse_against_noise(mse_list, mse_theory_list, target_labels, sv_overlaps, b)
    n = length(target_labels)
    fig = Figure(size = (400, 150 * n + 60))
    overlaps2 = abs2.(sv_overlaps.overlaps)
    c_mse, c_theory, c_sv = Makie.wong_colors()[1], Makie.wong_colors()[2], :steelblue
    exponents = floor(Int, log10(minimum(σE_list))):ceil(Int, log10(maximum(σE_list)))
    axes = map(enumerate(target_labels)) do (i, target)
        ax = Axis(fig[i, 1], ylabel = "MSE",
            title = "Estimating $(target[1]) ⊗ $(target[2])",
            xscale = log10, xticks = LogTicks(exponents), yticks = LinearTicks(4),
            xgridvisible = false, ygridvisible = false)
        vlines!(ax, sv_overlaps.vals .* sqrt(b),
            color = vec(overlaps2[i, :]), colormap = :blues,
            colorrange = (0, maximum(overlaps2)), linewidth = 1.5)
        lines!(
            ax, σE_list, getindex.(mse_theory_list, i), linewidth = 2.5, color = c_theory)
        scatter!(ax, σE_list, getindex.(mse_list, i), markersize = 7, color = c_mse)
        ax
    end
    linkxaxes!(axes...)
    hidexdecorations!.(axes[1:(end - 1)], ticks = false)
    axes[end].xlabel = "Noise level (σE)"
    Colorbar(fig[1:n, 2], colormap = :blues, limits = (0, maximum(overlaps2)),
        label = L"|(O|v_p)|²", ticks = LinearTicks(3))
    Legend(fig[n + 1, 1],
        [MarkerElement(color = c_mse, marker = :circle),
            LineElement(color = c_theory, linewidth = 2.5),
            LineElement(color = c_sv, linewidth = 1.5)],
        ["MSE", "Predicted MSE", "√b·σS"],
        orientation = :horizontal, framevisible = false)
    return fig
end
