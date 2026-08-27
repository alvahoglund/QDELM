
function effective_measurement(op, ρ_res, H_main, H_res, H_total)
    ρ_res_extend = embed(ρ_res, H_res => H_total)
    partial_trace(op * ρ_res_extend, H_total => H_main,
        alg = FermionicHilbertSpaces.FullPartialTraceAlg(); skipmissing = true)
end

function effective_measurement(op, ρ_res, qd_system::QuantumDotSystem)
    effective_measurement(
        op, ρ_res, qd_system.H_main, qd_system.H_res, qd_system.H_total)
end