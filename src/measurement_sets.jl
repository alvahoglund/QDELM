# =============================================================================
# measurement_sets.jl — single source of truth for the measurement choice. The same
# struct builds the operators (scrambling_map) and defines the row layout (noise models).
# Include after measurements.jl.
# =============================================================================
"""
Row layout of S: index = (t-1)*n_outcomes*M + n*M + j with outcome n = 0:n_outcomes-1,
dot j = 1:M (over `sys.grids.total`) and time t = 1:n_times. Rows (t, ·, j) form one POVM ("group").
"""
abstract type MeasurementSet end

"Complete charge measurement (0, 1, 2 electrons): the three rows of a group sum to the identity."
struct ChargeMeasurements012 <: MeasurementSet
    M::Int
    n_times::Int
end

"Charge measurement with outcomes 0, 1 only: not a complete POVM, no sum-zero constraint."
struct ChargeMeasurements01 <: MeasurementSet
    M::Int
    n_times::Int
end

_ntimes(::Number) = 1
_ntimes(ts) = length(ts)
ChargeMeasurements012(sys::QuantumDotSystem, ts) = ChargeMeasurements012(length(sys.grids.total), _ntimes(ts))
ChargeMeasurements01(sys::QuantumDotSystem, ts) = ChargeMeasurements01(length(sys.grids.total), _ntimes(ts))

n_outcomes(::ChargeMeasurements012) = 3
n_outcomes(::ChargeMeasurements01) = 2
nrows(ms::MeasurementSet) = n_outcomes(ms) * ms.M * ms.n_times

"Operators handed to `scrambling_map`; their order defines the row layout above."
operators(::ChargeMeasurements012, sys::QuantumDotSystem) = charge_probabilities(sys)
operators(::ChargeMeasurements01, sys::QuantumDotSystem) = charge_probabilities_01(sys)

function groups(ms::MeasurementSet)
    no = n_outcomes(ms)
    return [[(t - 1) * no * ms.M + n * ms.M + j for n in 0:(no - 1)]
            for t in 1:(ms.n_times) for j in 1:(ms.M)]
end

"Orthonormal basis (columns) of the space a group's noise lives in."
group_basis(::ChargeMeasurements012) = [1/√2 1/√6; -1/√2 1/√6; 0.0 -2/√6]
group_basis(::ChargeMeasurements01) = Matrix(1.0I, 2, 2)

function check_compatible(ms::MeasurementSet, sys::QuantumDotSystem, ts)
    ms.M == length(sys.grids.total) ||
        throw(ArgumentError("measurement set has M = $(ms.M) dots, system has $(length(sys.grids.total))"))
    ms.n_times == _ntimes(ts) ||
        throw(ArgumentError("measurement set has n_times = $(ms.n_times), got $(_ntimes(ts)) times"))
    return nothing
end

"Throw if S does not have the layout `ms` assumes."
function validate_layout(ms::MeasurementSet, S::AbstractMatrix; atol = 1e-8)
    size(S, 1) == nrows(ms) ||
        throw(DimensionMismatch("S has $(size(S, 1)) rows, measurement set expects $(nrows(ms))"))
    if ms isa ChargeMeasurements012
        d = isqrt(size(S, 2))
        id = vec(I(d))'
        for g in groups(ms)
            isapprox(sum(@view(S[g, :]), dims = 1), id; atol) ||
                throw(ArgumentError("rows $g of S do not sum to the identity; layout does not match ChargeMeasurements012"))
        end
    end
    return nothing
end
