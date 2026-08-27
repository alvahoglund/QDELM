
function make_test_system(; u_intra = 1.0, t = 1.0, t_so = 0.1, u_inter = 1.0)
    qd_system = tight_binding_system(2, 3, 1)
    pf = QDELM.random_param_functions(;
        u_intra = u_intra, t = t, t_so = t_so, u_inter = u_inter)
    hams = QDELM.matrix_representation_hams(hamiltonians(qd_system.grids, pf), qd_system)
    ψ_res = QDELM.eig_state(hams.res, 2)
    measurements = QDELM.charge_probabilities(qd_system)
    return qd_system, measurements, ψ_res, hams.total
end

function test_algorithms_agree(qd_system, measurements, ψ_res, ham_total, t; atol = 1e-4)
    S_diag = scrambling_map(qd_system, measurements, ψ_res, ham_total, t,
        QDELM.DiagonalizationPropagatorAlg())
    S_stepping = scrambling_map(qd_system, measurements, ψ_res, ham_total, t,
        QDELM.SteppingKrylovPropagatorAlg())
    S_adaptive = scrambling_map(qd_system, measurements, ψ_res, ham_total, t,
        QDELM.AdaptivePropagatorAlg())
    @test S_stepping≈S_diag atol=atol
    @test S_adaptive≈S_diag atol=atol
end

@testset "Scrambling map algorithms agree — default parameters, large t" begin
    qd_system, measurements, ψ_res, ham_total = make_test_system()
    test_algorithms_agree(qd_system, measurements, ψ_res, ham_total, 1000.0)
end

@testset "Scrambling map algorithms agree — large U_intra (large spectral radius)" begin
    qd_system, measurements, ψ_res, ham_total = make_test_system(u_intra = 100.0)
    test_algorithms_agree(qd_system, measurements, ψ_res, ham_total, 100.0)
end

@testset "Scrambling map algorithms agree — large t, multiple times" begin
    qd_system, measurements, ψ_res, ham_total = make_test_system()
    test_algorithms_agree(qd_system, measurements, ψ_res, ham_total, [200.0, 1000.0])
end

@testset "Scrambling map algorithms agree — large U_intra, multiple times" begin
    qd_system, measurements, ψ_res, ham_total = make_test_system(u_intra = 10.0)
    test_algorithms_agree(qd_system, measurements, ψ_res, ham_total, [100.0, 200.0])
end
