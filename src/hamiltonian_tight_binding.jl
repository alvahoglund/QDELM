struct DotParams{D1, D2, D3}
    ϵ::D1
    ϵb::D2
    u_intra::D3
end

struct InteractionParams{D1, D2, D3}
    t::D1
    t_so::D2
    u_inter::D3
end

struct ParamFunctions{F1, F2, F3, F4, F5, F6, F7}
    ϵ_func_main::F1
    ϵ_func_res::F2
    ϵb_func::F3
    u_intra_func::F4
    t_func::F5
    t_so_func::F6
    u_inter_func::F7
end

struct Hamiltonians{M, R, I, T, DM <: DotParams, DR <: DotParams, DI <: InteractionParams}
    main::M
    res::R
    intersection::I
    total::T
    dot_params_main::DM
    dot_params_res::DR
    interaction_params::DI
end

## ============ Singel dot ================

function hamiltonian_ϵ(ϵ, u_intra, grid)
    sum(
        (ϵ[coordinate] - u_intra[coordinate] / 2) * f[coordinate, σ]' * f[coordinate, σ]
        for σ in SPINS, coordinate in grid;
        init = 0
    )
end
function hamiltonian_b(ϵb, grid)
    sum(
        #1/2 as normalization factor for pauli matrices
        1 / 2 * ϵb[coordinate][1] *
        (f[coordinate, :↑]'f[coordinate, :↓] + f[coordinate, :↓]'f[coordinate, :↑]) + # Bx
        1 / 2 * ϵb[coordinate][2] *
        (-im * f[coordinate, :↑]'f[coordinate, :↓] +
         im * f[coordinate, :↓]'f[coordinate, :↑]) + #By
        1 / 2 * ϵb[coordinate][3] *
        (f[coordinate, :↑]'f[coordinate, :↑] - f[coordinate, :↓]'f[coordinate, :↓]) #Bz
        for coordinate in grid;
        init = 0
    )
end

function hamiltonian_c_intra(u_intra, grid)
    sum(
        u_intra[coordinate] *
        f[coordinate, :↑]'f[coordinate, :↓]'f[coordinate, :↓]f[coordinate, :↑]
        for coordinate in grid;
        init = 0
    )
end

## ============ Interactions ================

function hamiltonian_c_inter(u_inter, grid)
    hamiltonian_c_inter_x(u_inter, grid) +
    hamiltonian_c_inter_y(u_inter, grid)
end

function hamiltonian_c_inter_x(u_inter, grid)
    sum(
        u_inter[(i, j), (i + 1, j)] *
        f[(i, j), σ1]'f[(i + 1, j), σ2]'f[(i + 1, j), σ2]f[(i, j), σ1]
        for σ1 in SPINS, σ2 in SPINS, (i, j) in grid
        if (i + 1, j) in grid;
        init = 0
    )
end
function hamiltonian_c_inter_y(u_inter, grid)
    sum(
        u_inter[(i, j), (i, j + 1)] *
        f[(i, j), σ1]'f[(i, j + 1), σ2]'f[(i, j + 1), σ2]f[(i, j), σ1]
        for σ1 in SPINS, σ2 in SPINS, (i, j) in grid
        if (i, j + 1) in grid;
        init = 0
    )
end

function hamiltonian_t(t, grid)
    hamiltonian_t_x(t, grid) + hamiltonian_t_y(t, grid)
end

function hamiltonian_t_x(t, grid)
    sum(
        t[(i, j), (i + 1, j)]f[(i + 1, j), σ]'f[(i, j), σ] + hc
        for σ in SPINS, (i, j) in grid if (i + 1, j) in grid;
        init = 0
    )
end
function hamiltonian_t_y(t, grid)
    sum(
        t[(i, j), (i, j + 1)]f[(i, j + 1), σ]'f[(i, j), σ] + hc
        for σ in SPINS, (i, j) in grid if (i, j + 1) in grid;
        init = 0
    )
end

function hamiltonian_so(t_so, grid)
    hamiltonian_so_x(t_so, grid) +
    hamiltonian_so_y(t_so, grid)
end

function hamiltonian_so_x(t_so, grid)
    sum(
        t_so[(i, j), (i + 1, j)] *
        (-f[(i + 1, j), :↑]'f[(i, j), :↓] + f[(i + 1, j), :↓]'f[(i, j), :↑]) + hc
        for (i, j) in grid if (i + 1, j) in grid;
        init = 0
    )
end
function hamiltonian_so_y(t_so, grid)
    sum(
        t_so[(i, j), (i, j + 1)] *
        (im * f[(i, j + 1), :↑]'f[(i, j), :↓] + im * f[(i, j + 1), :↓]'f[(i, j), :↑]) + hc
        for (i, j) in grid if (i, j + 1) in grid;
        init = 0
    )
end

## ========= Set Dot Parameters =============
function ParamFunctions(;
        ϵ_func_main, ϵ_func_res, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
    ParamFunctions(
        ϵ_func_main, ϵ_func_res, ϵb_func, u_intra_func, t_func, t_so_func, u_inter_func)
end

function random_param_functions(; ϵ_main = 0.5, ϵ_res = 1, ϵb = [0, 0, 1],
        u_intra = 1.0, t = 1.0, t_so = 0.1, u_inter = 1.0)
    ParamFunctions(
        ϵ_func_main = () -> ϵ_main,
        ϵ_func_res = () -> rand() * ϵ_res,
        ϵb_func = () -> ϵb,
        u_intra_func = () -> u_intra * 10 + rand(),
        t_func = () -> rand() * t,
        t_so_func = () -> t_so * rand(),
        u_inter_func = () -> u_inter * rand()
    )
end

function equal_param_functions(;
        ϵ_main = 0.0, ϵ_res = 0.0, ϵb = [0, 0, 0], u_intra = 10.0, t = 1.0,
        t_so = 0.0, u_inter = 0.0)
    ParamFunctions(
        ϵ_func_main = () -> ϵ_main,
        ϵ_func_res = () -> ϵ_res,
        ϵb_func = () -> ϵb,
        u_intra_func = () -> u_intra,
        t_func = () -> t,
        t_so_func = () -> t_so,
        u_inter_func = () -> u_inter
    )
end

function set_dot_params(ϵ_func, ϵb_func, u_intra_func, grid)
    ϵ = Dict(coordinate => ϵ_func() for coordinate in grid)
    ϵb = Dict(coordinate => ϵb_func() for coordinate in grid)
    u_intra = Dict(coordinate => u_intra_func() for coordinate in grid)
    return DotParams(ϵ, ϵb, u_intra)
end

function set_interaction_params(t_func, t_so_func, u_inter_func, grid)
    coupled_coordinates = get_coupled_coordinates(grid)
    t = Dict(coupled_coordinate => t_func() for coupled_coordinate in coupled_coordinates)
    t_so = Dict(coupled_coordinate => t_so_func()
    for coupled_coordinate in coupled_coordinates)
    u_inter = Dict(coupled_coordinate => u_inter_func()
    for coupled_coordinate in coupled_coordinates)
    return InteractionParams(t, t_so, u_inter)
end

function get_coupled_coordinates(grid)
    coupled_coordinates_x = [((i, j), (i + 1, j))
                             for (i, j) in grid
                             if (i + 1, j) in grid]
    coupled_coordinates_y = [((i, j), (i, j + 1))
                             for (i, j) in grid
                             if (i, j + 1) in grid]
    return vcat(coupled_coordinates_x, coupled_coordinates_y)
end

## ========= System Hamiltonians =============

function hamiltonian_dots(dot_params, grid)
    hamiltonian_ϵ(dot_params.ϵ, dot_params.u_intra, grid) +
    hamiltonian_b(dot_params.ϵb, grid) +
    hamiltonian_c_intra(dot_params.u_intra, grid)
end

function hamiltonian_interactions(interaction_params, grid)
    hamiltonian_t(interaction_params.t, grid) +
    hamiltonian_so(interaction_params.t_so, grid) +
    hamiltonian_c_inter(interaction_params.u_inter, grid)
end

function hamiltonian_interactions_x(interaction_params, grid)
    hamiltonian_t_x(interaction_params.t, grid) +
    hamiltonian_so_x(interaction_params.t_so, grid) +
    hamiltonian_c_inter_x(interaction_params.u_inter, grid)
end

function hamiltonians(grids, seed = nothing)
    isnothing(seed) || Random.seed!(seed)
    hamiltonians(grids, random_param_functions())
end

function hamiltonians_equal_param(grids)
    hamiltonians(grids, equal_param_functions())
end

function hamiltonians(grids, pf::ParamFunctions)
    dot_params_main = set_dot_params(
        pf.ϵ_func_main, pf.ϵb_func, pf.u_intra_func, grids.main)
    dot_params_res = set_dot_params(pf.ϵ_func_res, pf.ϵb_func, pf.u_intra_func, grids.res)
    interaction_params = set_interaction_params(
        pf.t_func, pf.t_so_func, pf.u_inter_func, grids.total)
    hamiltonians(grids, dot_params_main, dot_params_res, interaction_params)
end

function hamiltonians(
        grids, dot_params_main::DotParams, dot_params_res::DotParams,
        interaction_params::InteractionParams)
    hamiltonian_main = hamiltonian_dots(dot_params_main, grids.main) +
                       hamiltonian_interactions(interaction_params, grids.main)
    hamiltonian_res = hamiltonian_dots(dot_params_res, grids.res) +
                      hamiltonian_interactions(interaction_params, grids.res)
    hamiltonian_intersection = hamiltonian_interactions_x(
        interaction_params, grids.intersection)
    hamiltonian_total = hamiltonian_main + hamiltonian_res + hamiltonian_intersection
    return Hamiltonians(hamiltonian_main, hamiltonian_res,
        hamiltonian_intersection, hamiltonian_total,
        dot_params_main, dot_params_res, interaction_params)
end

function matrix_representation_hams(hams::Hamiltonians, qd_system)
    Hamiltonians(
        hams.main,
        matrix_representation(hams.res, qd_system.H_res),
        hams.intersection,
        matrix_representation(hams.total, qd_system.H_total),
        hams.dot_params_main,
        hams.dot_params_res,
        hams.interaction_params
    )
end