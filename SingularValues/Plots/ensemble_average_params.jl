
function get_symlog_exp(sv_list_setting, sv_func)
    min_sv = minimum([minimum(filter(sv -> sv > 10^-8, sv_func.(sv_list)))
                      for sv_list in values(sv_list_setting)])
    return floor(log10(min_sv))
end

function plot_sv_against_param_range!(
        gl, sv_list_settings, parameter_range, title;
        axisledgend = true, xscale = log10, sv_func = minimum,
        yticks = nothing, symlog_exp = nothing, ylims = (nothing, nothing))
    if isnothing(symlog_exp)
        symlog_exp = get_symlog_exp(sv_list_settings, sv_func)
    end
    if isnothing(yticks)
        yticks = (vcat([0], [exp10(i) for i in range(symlog_exp, 1)]),
            vcat([L"0"], [L"10^{%$(i)}" for i in range(symlog_exp, 1)]))
    end
    ax = Axis(gl, xscale = xscale, xlabel = "Hamiltonian parameter",
        ylabel = "Singular values", title = title, yscale = Makie.Symlog10(10^symlog_exp),
        yticks = yticks)
    ylims!(ax, ylims[1], ylims[2])
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

function plot_sv_against_param_range!(
        gl, sv_list_params, parameter_range, title;
        axisledgend = true, xscale = log10, sv_func = minimum,
        yticks = nothing, symlog_exp = nothing, ylims = (nothing, nothing), legendposistion = :lt)
    if isnothing(symlog_exp)
        symlog_exp = get_symlog_exp(sv_list_params, sv_func)
    end
    if isnothing(yticks)
        yticks = (vcat([0], [exp10(i) for i in range(symlog_exp, 1)]),
            vcat([L"0"], [L"10^{%$(i)}" for i in range(symlog_exp, 1)]))
    end
    ax = Axis(gl, xscale = xscale, xlabel = "Hamiltonian parameter",
        ylabel = "Singular values", title = title, yscale = Makie.Symlog10(10^symlog_exp),
        yticks = yticks)
    ylims!(ax, ylims[1], ylims[2])
    for param in sort(collect(keys(sv_list_params)))
        sv_list = sv_list_params[param]
        avg_sv = sv_func.(sv_list)
        scatter!(ax, parameter_range, avg_sv, label = "$param")
        lines!(ax, parameter_range, avg_sv)
    end

    hlines!(
        ax, [10^symlog_exp], color = :grey, linestyle = :dash)
    vlines!(ax, [1], color = :grey, linestyle = :dash)
    if axisledgend
        axislegend(ax, position = legendposistion)
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