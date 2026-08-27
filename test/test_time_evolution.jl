
@testset "State and operator evolution" begin
    ρ = rand(ComplexF64, 4, 4) + hc
    ρ = ρ ./ tr(ρ)
    ham = (rand(ComplexF64, 4, 4) + hc) / 2
    op = (rand(ComplexF64, 4, 4) + hc) / 2
    t = 1.0

    ρt = state_time_evolution(ρ, t, ham)
    exp_value_ρ = expectation_value(ρt, op)

    op_t = operator_time_evolution(op, t, ham)
    exp_value_op = expectation_value(ρ, op_t)

    @test exp_value_op ≈ exp_value_ρ
end

@testset "Effective measurements & time evolution" begin
    qd_system = tight_binding_system(2, 3, 1)
    seed = 2
    hams = hamiltonians(qd_system.grids, seed)

    ham_res = matrix_representation(hams.res, qd_system.H_res)
    ψ_res = ground_state(ham_res)
    ρres = ψ_res * ψ_res'

    initial_states = map(density_matrix,
        [def_state(triplet_plus, qd_system.H_main),
            def_state(singlet, qd_system.H_main),
            random_product_state(qd_system),
            random_separable_state(3, qd_system)])

    total_states = map(
        initial_state -> tensor_product((initial_state, ρres),
            (qd_system.H_main, qd_system.H_res) => qd_system.H_total),
        initial_states)
    measurements = QDELM.charge_probabilities(qd_system)

    t = 10
    ham_total = matrix_representation(hams.total, qd_system.H_total)

    time_evolved_states = map(
        total_state -> state_time_evolution(total_state, t, ham_total), total_states)
    time_evolved_measurements = map(
        measurement -> operator_time_evolution(measurement, t, ham_total), measurements)
    effective_measurements = map(
        measurement -> effective_measurement(measurement, ρres, qd_system),
        time_evolved_measurements)
    scrambling_block = scrambling_map(
        qd_system, measurements, ψ_res, ham_total, t, QDELM.BlockPropagatorAlg())
    scrambling_pure = scrambling_map(
        qd_system, measurements, ψ_res, ham_total, t, QDELM.KrylovPropagatorAlg())

    @test scrambling_block ≈ scrambling_pure

    for i in eachindex(initial_states), j in eachindex(measurements)

        exp_val_1 = expectation_value(time_evolved_states[i], measurements[j])
        exp_val_2 = expectation_value(total_states[i], time_evolved_measurements[j])
        exp_val_3 = expectation_value(initial_states[i], effective_measurements[j])
        exp_val_4 = (scrambling_block * vec(initial_states[i]))[j]
        exp_val_5 = (scrambling_pure * vec(initial_states[i]))[j]
        @test exp_val_1 ≈ exp_val_2 ≈ exp_val_3 ≈ exp_val_4 ≈ exp_val_5
    end
end
