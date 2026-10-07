using QDELM, LinearAlgebra

nbr_dots_main = 2
nbr_dots_res = 3
qn_res = 2
sys = tight_binding_system(nbr_dots_main, nbr_dots_res, qn_res)
t = [100, 200, 300, 400, 500]
hams = matrix_representation_hams(hamiltonians(sys.grids), sys)
ψ_res = ground_state(hams.res)

ms01 = QDELM.ChargeMeasurements01(sys, t)
S01 = scrambling_map(sys, ms01, ψ_res, hams.total, t)

ms012 = QDELM.ChargeMeasurements012(sys, t)
S012 = scrambling_map(sys, ms012, ψ_res, hams.total, t)

Pm, _ = QDELM.pauli_matrix(sys.Hs_main, sys.H_main)
B = 1/2 * Pm[:, 2:end]
b = 0.0147

eigen(QDELM.information_matrix(QDELM.NaiveNoise(1), S01, B, b)).values ≈
eigen(QDELM.information_matrix(QDELM.NaiveNoise(1), S012, B, b)).values #False

eigen(QDELM.information_matrix(QDELM.NaiveNoise(1), S012, B, b)).values .* 2/3 ≈
real.(eigen(QDELM.information_matrix(QDELM.IsotropicNoise(1, ms01), S01, B, b)).values) #True

real.(eigen(QDELM.information_matrix(QDELM.IsotropicNoise(1, ms01), S01, B, b)).values) ≈
real.(eigen(QDELM.information_matrix(QDELM.IsotropicNoise(1, ms012), S012, B, b)).values) #True

eigen(QDELM.information_matrix(QDELM.ShotNoise(1, ms01), S01, B, b)).values ≈
real.(eigen(QDELM.information_matrix(QDELM.ShotNoise(1, ms012), S012, B, b)).values) #True

