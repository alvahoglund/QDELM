const f = FermionicHilbertSpaces.SymbolicFermionBasis(:f, 0)
const SPINS = (:↑, :↓)
struct Grids{M, R, T, I}
    main::M
    res::R
    total::T
    intersection::I
end

struct QuantumDotSystem{G, HsM, HM, HR, HT}
    grids::G

    Hs_main::HsM
    H_main::HM

    H_res::HR
    H_total::HT
end

function labels(coordinates)
    [(coordinate, spin) for coordinate in coordinates for spin in SPINS]
end

sites(H) = unique(first.(keys(H)))

function tight_binding_system(nbr_dots_main::Number, nbr_dots_res::Number, qn_res::Number)
    grids = generate_grid(nbr_dots_main, nbr_dots_res)
    tight_binding_system(grids, qn_res)
end

function tight_binding_system(grids::Grids, qn_res)
    qn_total = qn_res + length(grids.main)

    Hs_main = [hilbert_space(labels((coordinate,)), NumberConservation(1))
               for coordinate in grids.main]
    H_main = tensor_product(Hs_main)
    H_res = hilbert_space(labels(grids.res), NumberConservation(qn_res))

    H_total = hilbert_space(labels(grids.total), NumberConservation(qn_total))

    QuantumDotSystem(grids, Hs_main, H_main, H_res, H_total)
end

function generate_grid(nbr_dots_main::Int, nbr_dots_res::Int)
    grid_main = [(1, i) for i in 1:nbr_dots_main]
    grid_res = [(div(i - 1, nbr_dots_main) + 2, mod1(i, nbr_dots_main))
                for i in 1:nbr_dots_res]
    grid_total = vcat(grid_main, grid_res)
    grid_intersection = vcat(grid_main,
        [coordinate for coordinate in grid_res if coordinate[1] == 2])
    return Grids(grid_main, grid_res, grid_total, grid_intersection)
end

qn_sector(H::SymmetricFockHilbertSpace) = H.symmetry.conserved_quantity.sectors[1]