
@testset "Hamiltonian of tight binding model" begin
    qd_system = tight_binding_system(2, 3, 1)
    hams = hamiltonians(qd_system.grids)
    h_main = matrix_representation(hams.main, qd_system.H_total)
    h_res = matrix_representation(hams.res, qd_system.H_total)
    h_int = matrix_representation(hams.intersection, qd_system.H_total)
    h_tot = matrix_representation(hams.total, qd_system.H_total)
    @test h_main' ≈ h_main
    @test h_res' ≈ h_res
    @test h_int' ≈ h_int
    @test h_tot' ≈ h_tot
end

@testset "Sum of Hamiltonians" begin
    # Check that H_main + H_res + H_intersection = H_total_system    
    qd_system = tight_binding_system(3, 5, 2)
    ϵ_func() = 2
    ϵb_func() = [0, 0, 1]
    u_intra_func() = 10

    t_func() = 1
    t_so_func() = 0.1
    u_inter_func() = 1

    main_system_dot_params = set_dot_params(
        ϵ_func, ϵb_func, u_intra_func, qd_system.grids.main)
    res_dot_params = set_dot_params(ϵ_func, ϵb_func, u_intra_func, qd_system.grids.res)
    total_dot_params = set_dot_params(ϵ_func, ϵb_func, u_intra_func, qd_system.grids.total)
    interaction_params = set_interaction_params(
        t_func, t_so_func, u_inter_func, qd_system.grids.total)

    hams_sum = hamiltonians(
        qd_system.grids, main_system_dot_params, res_dot_params, interaction_params)

    ham_total = hamiltonian_dots(total_dot_params, qd_system.grids.total) +
                hamiltonian_interactions(
        interaction_params, qd_system.grids.total)

    @test matrix_representation(hams_sum.total, qd_system.H_total) ≈
          matrix_representation(ham_total, qd_system.H_total)
end
