function plot_heatmap_W_spin_basis!(gl, W_eff)
    ax = Axis(gl[1, 1])
    hm = heatmap!(
        ax, 1:4, 1:4, Matrix(reshape(QDELM.to_real.(W_eff), (4, 4))),
        colormap = :bam, colorrange = [-1, 1])

    ax.xticks = (1:4, ["σ0", "σx", "σy", "σz"])
    ax.yticks = (1:4, ["σ0", "σx", " σy", "σz"])
    return hm
end

function plot_linear_ew!(gl, W_eff_sub, Ω_sub_sep, Ω_sub_ent;
        step_size_sep = 1, step_size_ent = 80, title = "Linear Decision Boundary")
    ax = Axis3(gl[1, 1], xlabel = "⟨σx⊗σx⟩", ylabel = "⟨σy⊗σy⟩", zlabel = "⟨σz⊗σz⟩",
        title = title)
    scatter!(ax,
        Ω_sub_sep[1:step_size_sep:end, 2],
        Ω_sub_sep[1:step_size_sep:end, 3],
        Ω_sub_sep[1:step_size_sep:end, 4],
        label = "Separable", markersize = 5
    )
    scatter!(ax,
        Ω_sub_ent[1:step_size_ent:end, 2],
        Ω_sub_ent[1:step_size_ent:end, 3],
        Ω_sub_ent[1:step_size_ent:end, 4],
        label = "Entangled", markersize = 5
    )

    x_range, y_range, db_plane = get_linear_db(W_eff_sub)
    surface!(ax, x_range, y_range, db_plane)
end

function get_linear_db(W_spin)
    x_range = range(-1, 1, length = 100)
    y_range = range(-1, 1, length = 100)
    b = QDELM.to_real(W_spin[1])
    wx = QDELM.to_real(W_spin[2])
    wy = QDELM.to_real(W_spin[3])
    wz = QDELM.to_real(W_spin[4])
    db_plane = [-(b + wx * x + wy * y) / wz for x in x_range, y in y_range]
    return x_range, y_range, db_plane
end

function plot_nonlinear_db_spin_space!(
        gl, Ω_sub_sep, Ω_sub_ent, W, S, X_train_mean, feature_transformation_alg;
        n_grid = 25, title = "Nonlinear decision boundary")
    Pm_sub = get_sub_spin_basis(sys.Hs_main, sys.H_main)
    grid = range(-1, 1, length = n_grid)
    step_sep = 1
    step_ent = 100
    function eval_point(xx, yy, zz)
        X_vec = QDELM.to_real.(S * (Pm_sub * [1.0, xx, yy, zz] ./ 4))
        Z_vec = QDELM.add_bias(reshape(X_vec .- X_train_mean, :, 1))
        X_poly = QDELM.feature_transformation(Z_vec, feature_transformation_alg)
        Float32(dot(vec(X_poly), W))
    end
    vals = Float32[eval_point(xx, yy, zz) for xx in grid, yy in grid, zz in grid]

    ax = Axis3(gl[1, 1], xlabel = "⟨σx⊗σx⟩", ylabel = "⟨σy⊗σy⟩", zlabel = "⟨σz⊗σz⟩",
        title = title)

    scatter!(ax, Ω_sub_sep[1:step_sep:end, 2],
        Ω_sub_sep[1:step_sep:end, 3], Ω_sub_sep[1:step_sep:end, 4],
        label = "Separable", markersize = 5)
    scatter!(
        ax, Ω_sub_ent[1:step_ent:end, 2],
        Ω_sub_ent[1:step_ent:end, 3], Ω_sub_ent[1:step_ent:end, 4],
        label = "Entangled", markersize = 5)

    volume!(ax, (-1, 1), (-1, 1), (-1, 1), vals;
        algorithm = :iso, isovalue = 0.0f0, isorange = 0.1f0, alpha = 0.5)

    Legend(gl[2, 1], ax, orientation = :horizontal, framevisible = false)
end

function plot_nonlinear_db_spin_space(
        Ω_sub_sep, Ω_sub_ent, W, S, X_train_mean, feature_transformation_alg; n_grid = 25)
    fig = Figure()
    plot_nonlinear_db_spin_space!(
        fig, Ω_sub_sep, Ω_sub_ent, W, S, X_train_mean, feature_transformation_alg; n_grid)
    display(fig)
end