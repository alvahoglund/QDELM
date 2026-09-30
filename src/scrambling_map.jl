# =============================================================================
# scrambling_map.jl
#
# Enclosing module (QDELM) must have:
#     using LinearAlgebra, SparseArrays
# and QDELM's own: dim, generalized_kron, density_matrix, to_dense, propagator,
# operator_time_evolution, effective_measurement, QuantumDotSystem.
#
# Two ways to compute the scrambling map:
#   - Operator path (BlockPropagatorAlg): builds the full propagator U and evolves
#     the measurement operators. Allows a mixed reservoir state; expensive.
#   - State path (AbstractStatePropagatorAlg): propagates the N×N_main block
#     Ψ0 = [e_j ⊗ ψres]_j. Pure reservoir state only. Its central primitive is
#         Us, info = propagate_block(H, Ψ0, ts, alg)
#     with Us[j] ≈ exp(-i H ts[j]) Ψ0 and info::NamedTuple containing at least
#     (method, err_est, nmatvec); err_est = NaN / nmatvec = -1 when unavailable.
#
# State-path algorithms:
#   here                       DiagonalizationPropagatorAlg
#   scrambling_map_qp.jl       QPAlg, AutoPropagatorAlg
#   scrambling_map_legacy.jl   older implementations, kept for benchmarking
# =============================================================================
abstract type AbstractPropagatorAlg end

# -----------------------------------------------------------------------------
# Operator path: full propagator U — allows mixed reservoir state
# -----------------------------------------------------------------------------
struct BlockPropagatorAlg <: AbstractPropagatorAlg end

function scrambling_map(H_main, H_res, H_total, measurements, ψres, hamiltonian, t::Number, ::BlockPropagatorAlg)
    ρ_res = density_matrix(ψres)
    U = propagator(t, hamiltonian)
    measurements_t = map(Base.Fix1(operator_time_evolution, U), measurements)
    eff_measurements = map(
        mt -> effective_measurement(mt, ρ_res, H_main, H_res, H_total),
        measurements_t)
    return reduce(vcat, (vec(m)' for m in eff_measurements))
end

function scrambling_map(H_main, H_res, H_total, measurements, ψres, hamiltonian, ts::AbstractArray, alg::BlockPropagatorAlg)
    mapreduce(t -> scrambling_map(H_main, H_res, H_total, measurements, ψres,
        hamiltonian, t, alg), vcat, ts)
end

# -----------------------------------------------------------------------------
# Pure state block propagator algorithms - only allows pure reservoir state
# -----------------------------------------------------------------------------
"Algorithms that propagate the pure-state block Ψ0 = [e_j ⊗ ψres]_j."
abstract type AbstractStatePropagatorAlg <: AbstractPropagatorAlg end

_tvec(t::Real) = [t]
_tvec(ts) = collect(ts)
_increments(ts::Vector{T}) where T = isempty(ts) ? T[] : [ts[1]; diff(ts)]

_operator(H::SparseMatrixCSC) = H
_operator(H::StridedMatrix{<:Complex}) = H
_operator(H::StridedMatrix{R}) where R<:Real = Matrix{complex(R)}(H)   # keep BLAS on complex blocks
_operator(H::AbstractMatrix) = sparse(H)
_operator(H) = H

_nnz_per_row(H::SparseMatrixCSC) = nnz(H) / size(H, 1)
_nnz_per_row(H) = float(size(H, 2))

"""
    initial_block(H_main, H_res, H_total, ψres) -> Ψ0 :: Matrix{ComplexF64} (N × N_main)
Columns are e_j ⊗ ψres embedded in the total Hilbert space.
"""
function initial_block(H_main, H_res, H_total, ψres::AbstractVector{T}) where T
    N_main = dim(H_main)
    e_j = zeros(T, N_main)
    stack(1:N_main) do n
        fill!(e_j, 0)
        e_j[n] = 1
        generalized_kron((e_j, ψres), (H_main, H_res) => H_total)
    end
end

"Rows vec(U' D_k U)' for each (diagonal) measurement D_k — same convention as before."
measurement_rows(U, measurements) = stack(op -> vec(U' * (Diagonal(op) * U)), measurements)'

function scrambling_map(H_main, H_res, H_total, measurements, ψres::AbstractVector,
    hamiltonian, t::Number, alg::AbstractStatePropagatorAlg)
    Ψ0 = initial_block(H_main, H_res, H_total, ψres)
    Us, _ = propagate_block(hamiltonian, Ψ0, _tvec(t), alg)
    return measurement_rows(only(Us), measurements)
end

function scrambling_map(H_main, H_res, H_total, measurements, ψres::AbstractVector,
    hamiltonian, ts::AbstractVector{<:Real}, alg::AbstractStatePropagatorAlg)
    Ψ0 = initial_block(H_main, H_res, H_total, ψres)
    Us, _ = propagate_block(hamiltonian, Ψ0, ts, alg)
    return reduce(vcat, [measurement_rows(U, measurements) for U in Us])
end

# -----------------------------------------------------------------------------
# Diagonalization algorithm 
# -----------------------------------------------------------------------------
"""
    DiagonalizationPropagatorAlg()
Dense Hermitian eigendecomposition once; every time point is then two GEMMs.
"""
struct DiagonalizationPropagatorAlg <: AbstractStatePropagatorAlg end

function propagate_block(H, Ψ0, ts, ::DiagonalizationPropagatorAlg)
    tsv = _tvec(ts)
    F = eigen(Hermitian(to_dense(H)))
    C = F.vectors' * Ψ0
    tmp = similar(C, complex(eltype(C)))
    Us = map(tsv) do t
        tmp .= cis.(-t .* F.values) .* C
        F.vectors * tmp
    end
    tmax = isempty(tsv) ? 0.0 : maximum(abs, tsv)
    err = eps() * size(H, 1) * maximum(abs, F.values) * max(1.0, tmax)   # heuristic
    return Us, (; method=:diagonalization, err_est=err, nmatvec=0)
end

# -----------------------------------------------------------------------------
# System-level entry point (default is AutoPropagatorAlg)
# -----------------------------------------------------------------------------
function scrambling_map(sys::QuantumDotSystem, measurements, ψres,
    hamiltonian, t, alg=AutoPropagatorAlg())
    scrambling_map(sys.H_main, sys.H_res, sys.H_total, measurements, ψres,
        hamiltonian, t, alg)
end
