## Standard time evolution
propagator(t, hamiltonian) = cis(-t * Matrix(hamiltonian))

operator_time_evolution(propagator, operator) = propagator' * operator * propagator
function operator_time_evolution(operator, t, hamiltonian)
    operator_time_evolution(propagator(t, hamiltonian), operator)
end

state_time_evolution(propagator, ρ) = propagator * ρ * propagator'
function state_time_evolution(ρ, t, hamiltonian)
    state_time_evolution(propagator(t, hamiltonian), ρ)
end
