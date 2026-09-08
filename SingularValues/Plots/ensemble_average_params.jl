
function get_symlog_exp(sv_list_setting, sv_func)
    min_sv = minimum([minimum(filter(sv -> sv > 10^-10, sv_func.(sv_list)))
                      for sv_list in values(sv_list_setting)])
    return floor(log10(min_sv))
end

function plot_sv_against_param_range!(
        gl, sv_list_settings, parameter_range, title;
        axisledgend = true, xscale = log10, sv_func = minimum,
        yticks = nothing, symlog_exp = nothing)
    if isnothing(symlog_exp)
        symlog_exp = get_symlog_exp(sv_list_settings, sv_func)
    end
    if isnothing(yticks)
        yticks = ([0, 10^(symlog_exp), 10^(-3), 10^(-2), 10^(-1), 1, 10],
            [L"0", L"10^{%$(symlog_exp)}", L"10^{-3}", L"10^{-2}", L"10^{-1}", L"1", L"10"])
    end

    ax = Axis(gl, xscale = xscale, xlabel = "Hamiltonian parameter",
        ylabel = "Singular values", title = title, yscale = Makie.Symlog10(10^symlog_exp), yticks = yticks)
    for setting in sort(collect(keys(sv_list_settings)))
        sv_list = sv_list_settings[setting]
        avg_sv = sv_func.(sv_list)
        scatter!(ax, parameter_range, avg_sv, label = "res = $(setting[1]), qn = $(setting[2])")
        lines!(ax, parameter_range, avg_sv)
    end

    hlines!(
        ax, [10^symlog_exp], color = :grey, linestyle = :dash, label = "linear scale limit")
    vlines!(ax, [1], color = :grey, linestyle = :dash, label = "1")
    if axisledgend
        axislegend(ax, position = :lt)s
    end
end

function add_legend_settings!(gl, settings)
    ax = Axis(gl[1, 1])
    hidedecorations!(ax)
    hidespines!(ax)
    for (res, qn) in settings
        scatter!(ax, [NaN], [NaN], label = "res = $res, qn = $qn")
        lines!(ax, [NaN], [NaN])
    end
    axislegend(ax, position = :lt)
end
function condition_number(sv)
    maximum(sv) / minimum(sv)
end