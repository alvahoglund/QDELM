using QDELM
using JLD2
## =================== System A ======================
nbr_dots_res = 6
qn_res = 3
sys = tight_binding_system(2, nbr_dots_res, qn_res)
seed = 1323
hams = QDELM.matrix_representation_hams(QDELM.hamiltonians(sys.grids, seed), sys)
ρ_res = ground_state(hams.res)
t_list = [100, 200]
measurements = QDELM.charge_probabilities(sys)
S = scrambling_map(sys, measurements, ρ_res, hams.total, t_list)

save("DefaultSystems/scrambling_map_A.jld2", "S", S, "t_list", t_list,
    "hams", hams, "measurements", measurements, "sys", sys)

## ================== System B ======================
nbr_dots_res = 4
qn_res = 1
sys = tight_binding_system(2, nbr_dots_res, qn_res)
seed = 39212
hams = QDELM.matrix_representation_hams(QDELM.hamiltonians(sys.grids, seed), sys)
ρ_res = ground_state(hams.res)
t_list = [100, 200, 300]
measurements = QDELM.charge_probabilities(sys)
S = scrambling_map(sys, measurements, ρ_res, hams.total, t_list)
save("DefaultSystems/scrambling_map_B.jld2", "S", S, "t_list", t_list,
    "hams", hams, "measurements", measurements, "sys", sys)

## ================== System C ======================
nbr_dots_res = 4
qn_res = 1
sys = tight_binding_system(2, nbr_dots_res, qn_res)
seed = 39212
hams = QDELM.matrix_representation_hams(QDELM.hamiltonians(sys.grids, seed), sys)
ρ_res = ground_state(hams.res)
t_list = [100, 200, 300]
measurements = QDELM.charge_probabilities_01(sys)
S = scrambling_map(sys, measurements, ρ_res, hams.total, t_list)
save("DefaultSystems/scrambling_map_C.jld2", "S", S, "t_list", t_list,
    "hams", hams, "measurements", measurements, "sys", sys)

## ================== System D ======================
nbr_dots_res = 6
qn_res = 3
sys = tight_binding_system(2, nbr_dots_res, qn_res)
seed = 1323
hams = QDELM.matrix_representation_hams(QDELM.hamiltonians(sys.grids, seed), sys)
ρ_res = ground_state(hams.res)
t_list = [100, 200]
measurements = QDELM.charge_probabilities_01(sys)
S = scrambling_map(sys, measurements, ρ_res, hams.total, t_list)

save("DefaultSystems/scrambling_map_D.jld2", "S", S, "t_list", t_list,
    "hams", hams, "measurements", measurements, "sys", sys)
