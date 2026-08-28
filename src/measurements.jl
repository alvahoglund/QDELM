## ============ Measurements =================
σ0(i) = (f[i, :↑]' * f[i, :↑] + f[i, :↓]' * f[i, :↓])
σx(i) = (f[i, :↓]' * f[i, :↑] + f[i, :↑]' * f[i, :↓])
σy(i) = im * (f[i, :↓]' * f[i, :↑] - f[i, :↑]' * f[i, :↓])
σz(i) = (f[i, :↑]' * f[i, :↑] - f[i, :↓]' * f[i, :↓])

function nbr_op(coordinate)
    f[coordinate, :↑]' * f[coordinate, :↑] + f[coordinate, :↓]' * f[coordinate, :↓]
end
function nbr2_op(coordinate)
    f[coordinate, :↑]' * f[coordinate, :↑] * f[coordinate, :↓]' * f[coordinate, :↓]
end

p(nbr_index, coordinate) = eval(Expr(:call, Symbol("p", nbr_index), coordinate))
p0(coordinate) = 1 - p1(coordinate) - p2(coordinate) #Probability to measure 0 charge
p1(coordinate) = nbr_op(coordinate) - 2 * nbr2_op(coordinate) # Probability to measure 1 charge
p2(coordinate) = nbr2_op(coordinate) # Probability to measure 2 charges

const PauliKeys = (:σ0, :σx, :σy, :σz)
function paulis(H, Hfinal = H)
    dim(H) == 2 ||
        throw(ArgumentError("Paulis is only defined for 2-dimensional Hilbert spaces"))
    coords = sites(H)
    x = only(coords)
    I = embed([1 0; 0 1], H => Hfinal)
    X = embed([0 1; 1 0], H => Hfinal)
    Y = embed([0 -im; im 0], H => Hfinal)
    Z = embed([1 0; 0 -1], H => Hfinal)
    vals = [I, X, Y, Z]
    return Dict(map(Pair, PauliKeys, vals))
end
function pauli_strings(Hs, Hfinal)
    ps = map(H -> collect(paulis(H, Hfinal)), Hs)
    pairs = map(Iterators.product(ps...)) do pairs
        key = tuple(first.(pairs)...)
        mat = prod(last.(pairs))
        key => mat
    end
    Dict(pairs)
end
function pauli_matrix(Hs, Hfinal)
    # each column of P is a vectorized Pauli string
    ps = pauli_strings(Hs, Hfinal)
    P = stack(vec, ps[a, b] for a in PauliKeys for b in PauliKeys)
    pauli_indices = Dict((a, b) => i
    for (i, (a, b)) in enumerate(Iterators.product(PauliKeys, PauliKeys)))
    return P, pauli_indices
end

function to_real(value, tolerance = 1e-3)
    abs(imag(value)) < tolerance ? real(value) :
    throw(ArgumentError("The value has an imaginary part: $(imag(value))"))
end
expectation_value(ρ, op::AbstractMatrix) = to_real(tr(density_matrix(ρ) * op))
variance(ρ, op) = expectation_value(ρ, op^2) - expectation_value(ρ, op)^2

## ======== Measurement sets =================
zero_charge_probabilities(grid) = map(p0, grid)
single_charge_probabilities(grid) = map(p1, grid)
double_charge_probabilities(grid) = map(p2, grid)

function charge_probabilities(grid)
    vcat(zero_charge_probabilities(grid), single_charge_probabilities(grid),
        double_charge_probabilities(grid))
end

single_charge_measurements(grid) = map(nbr_op, grid)
double_charge_measurements(grid) = map(nbr2_op, grid)

function charge_measurements(grid)
    vcat(single_charge_measurements(grid), double_charge_measurements(grid))
end

matrix_representation_ops(ops, H) = map(Base.Fix2(matrix_representation, H), ops)

function charge_measurements(qd_system::QuantumDotSystem)
    matrix_representation_ops(
        charge_measurements(qd_system.grids.total), qd_system.H_total)
end
function charge_probabilities(qd_system::QuantumDotSystem)
    matrix_representation_ops(
        charge_probabilities(qd_system.grids.total), qd_system.H_total)
end

function correlated_measurements(grid, qn_total)
    valid_combos = get_measurement_combinations(grid, qn_total)
    measurement_ops = map(Base.Fix1(measurement_combination_op, grid), valid_combos)
    return measurement_ops
end
function get_measurement_combinations(grid, qn_total)
    nbr_coordinates = length(grid)
    all_combos = Iterators.product(ntuple(_ -> 0:2, nbr_coordinates)...)
    valid_combos = [measurement_combo
                    for measurement_combo in all_combos
                    if qn_total == sum(measurement_combo)]
    return valid_combos
end
function measurement_combination_op(grid, measurement_combo)
    prod([p(measurement_combo[i], coord) for (i, coord) in enumerate(grid)])
end
function correlated_measurements(qd_system)
    matrix_representation_ops(
        correlated_measurements(
            qd_system.grids.total, qn_sector(qd_system.H_total)), qd_system.H_total)
end

## ============= Spin measurements ======================

#Operator for total spin S^2 on coodinate i 
Si2(coordinate_i, H_i) = matrix_representation(3 / 4 * p1(coordinate_i), H_i)

# Operator for S_i ⋅ S_j
function Sij(coordinate_i, coordinate_j, H)
    Hi = hilbert_space(labels((coordinate_i,)), NumberConservation(1))
    Hj = hilbert_space(labels((coordinate_j,)), NumberConservation(1))
    ps = pauli_strings((Hi, Hj), H)
    1 / 4 * sum(ps[σ, σ] for σ in [:σx, :σy, :σz])
end

# S^2 operator 
function total_spin_op(coordinates, H)
    S2_op = sum([Si2(coordinate, H) for coordinate in coordinates])
    Sij_op = sum(
        (Sij(coordinate_i, coordinate_j, H)
        for (i, coordinate_i) in enumerate(coordinates)
        for (j, coordinate_j) in enumerate(coordinates)
        if i < j),
        init = zero(S2_op))
    return S2_op + 2 * Sij_op
end

#S^2 = S(S+1) if the state is an eigenstate of S^2
s_from_s2(s2_val) = -1 / 2 + √(s2_val + 1 / 4)

## ================= Add noise to measurements =====================
function add_noise(X, σE)
    E = rand(Normal(0, σE), size(X))
    return X + E
end