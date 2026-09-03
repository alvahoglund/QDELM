
function plot_sv_m!(pl, sv_mat, mean_sv, M_max; title = "")
    ax = Axis(
        pl[1, 1], yscale = Makie.Symlog10(1e-6), ylabel = "Singular values",
        xlabel = "Number of multiplexed  \n scrambling maps", title = title, xscale = log10,
        yticks = ([0, 1e-5, 1e-4, 1e-3, 1e-2],
            ["0", "10⁻⁵", "10⁻⁴", "10⁻³", "10⁻²"]))
    for j in axes(sv_mat, 1)
        lines!(ax, 1:M_max, sv_mat[j, :], color = :steelblue)
        scatter!(ax, 1:M_max, sv_mat[j, :], color = :steelblue, markersize = 3)
    end
    lines!(ax, 1:M_max, mean_sv, color = :orange, linewidth = 2)
    elements = [MarkerElement(color = :steelblue, marker = :circle),
        LineElement(color = :orange, linewidth = 2)]
    labels = ["Singular values", "Mean singular value"]
    axislegend(ax, elements, labels, position = :rb)
end

function plot_κ_m!(pl, κ, M_max; title = "")
    ax = Axis(
        pl[1, 1], yscale = log10, ylabel = "Condition number",
        xlabel = "Number of multiplexed  \n scrambling maps", title = title, xscale = log10)
    lines!(ax, 1:M_max, κ, color = :orange, label = "Condition number")
    axislegend(ax, position = :rt)
end

function plot_sv_lines!(pl, svs; title = "")
    ax = Axis(pl[1, 1], yscale = log10, ylabel = "Singular values", title = title)
    ylims!(ax, 10^(-3.5), nothing)
    hlines!(ax, svs)
end

## Plot singular vectors
function pauli_grid(v, j)
    [(a == 1 && b == 1) ? NaN : v[(a - 1) * 4 + b - 1, j] for a in 1:4, b in 1:4]
end

function plot_pauli_overlaps!(gl, F; ncols = 5, title = "")
    overlap = abs2.(F.V)
    n = size(overlap, 2)
    nrows = ceil(Int, n / ncols)
    vmax = maximum(filter(!isnan, overlap))
    pauli_tick_labels = collect(string.(QDELM.PauliKeys))
    local hm
    for j in 1:n
        row, col = fldmod1(j, ncols)
        ax = Axis(gl[row, col], aspect = 1, title = "j = $j",
            xticks = (1:4, pauli_tick_labels), yticks = (1:4, pauli_tick_labels))
        hm = heatmap!(
            ax, pauli_grid(overlap, j), colormap = :Blues, colorrange = (0, vmax),
            nan_color = :white)
    end
    Colorbar(gl[1:nrows, ncols + 1], hm, label = "Overlap |⟨Normalized Pauli string|singular vector⟩|²")
    Label(gl[nrows + 1, 1:ncols], title, fontsize = 20, font = :bold)
end

