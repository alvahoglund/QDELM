module QDELM
using LinearAlgebra, Random, Arpack, SparseArrays, LinearMaps, Statistics
using ExponentialUtilities
using Reexport
@reexport using FermionicHilbertSpaces
@reexport using FermionicHilbertSpaces: indices, sector, AbstractHilbertSpace,
                                        NonCommutativeProducts

using Distributions: Normal
using ArnoldiMethod, KrylovKit

export tight_binding_system, hamiltonians, hamiltonian_dots, hamiltonian_interactions
export random_separable_state, random_product_state, triplet_plus, singlet, ground_state,
       def_state, density_matrix, matrix_representation_hams
export charge_measurements, effective_measurement, scrambling_map, expectation_value,
       charge_probabilities, matrix_representation_ops, pauli_strings, pauli_matrix
export state_time_evolution, operator_time_evolution
export set_dot_params, set_interaction_params

include("quantum_dot_system.jl")
include("hamiltonian_tight_binding.jl")
include("states.jl")
include("time_evolution.jl")
include("measurements.jl")
include("effective_measurements.jl")
include("scrambling_map.jl")
include("regression.jl")

end