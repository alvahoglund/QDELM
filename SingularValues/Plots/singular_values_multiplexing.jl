## ======================= Ensemble average varying qn =========================
condition_number(sv) = maximum(sv) / minimum(sv)

function plot_sv_qn!(gl, ens_sv_res, sv_func; title = "",
        ylabel = "Singular values", legend_position = :rb)
    ax = Axis(gl[1, 1], xlabel = "Res electrons relative to half filling",
        ylabel = ylabel, title = title, xscale = identity, yscale = log10)
    xmax = 0
    for nbr_dots_res in sort(collect(keys(ens_sv_res)))
        sv_list = ens_sv_res[nbr_dots_res]
        avg_sv = sv_func.(sv_list)
        x_vals = range(-length(avg_sv)÷2, length(avg_sv) ÷ 2, length = length(avg_sv))
        scatter!(ax, x_vals, avg_sv, markersize = 8,
            label = "Res dots: $nbr_dots_res")
        lines!(ax, x_vals, avg_sv)
        xmax = max(xmax, length(avg_sv)÷2)
    end
    ax.xticks = (-xmax):xmax
    axislegend(ax, position = legend_position)
end
function plot_sv_stats_qn(gl, ens_sv_res; title = "")
    Label(gl[0, 1:4], title, fontsize = 20)
    plot_sv_qn!(gl[1, 1], ens_sv_res, minimum;
        title = "Minimum singular value of ensemble average",
        ylabel = "Minimum singular value")
    plot_sv_qn!(gl[1, 2], ens_sv_res, mean;
        title = "Mean singular value of ensemble average", ylabel = "Mean singular value")
    plot_sv_qn!(gl[1, 3], ens_sv_res, median;
        title = "Median singular value of ensemble average", ylabel = "Median singular value")
    plot_sv_qn!(gl[1, 4], ens_sv_res, condition_number;
        title = "Condition number of ensemble average", ylabel = "Condition number", legend_position = :rb)
    gl
end
