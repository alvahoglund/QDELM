@testset "Eigenstates" begin
    Nmain = 2
    Nres = 2
    qn_res = 2
    qn_total = 2 + 2
    sys = tight_binding_system(Nmain, Nres, qn_res)
    hams_symbolic = hamiltonians(sys.grids)
    hamiltonian_total = matrix_representation(hams_symbolic.total, sys.H_total)

    ψ_arnoldi = ground_state(hamiltonian_total, QDELM.ArnoldiAlg())
    ψ_exact = ground_state(hamiltonian_total, QDELM.ExactDiagonalizationAlg())
    ψ_krylov = ground_state(hamiltonian_total, QDELM.KrylovAlg())

    @test(norm(ψ_arnoldi)≈norm(ψ_exact)≈norm(ψ_krylov)≈1.0)
    @test isapprox(abs(ψ_arnoldi' * ψ_exact), 1.0; atol = 1e-7)
    @test isapprox(abs(ψ_krylov' * ψ_exact), 1.0; atol = 1e-7)
end