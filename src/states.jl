## =================== Vaccum state ================ 
function vac_state(H)
    v0 = spzeros(dim(H))
    v0[FermionicHilbertSpaces.state_index(FockNumber(0), H)] = 1.0
    return v0
end

## =================== 2 dot main system states =============
function singlet()
    1 / √2 * (f[(1, 1), :↑]' * f[(1, 2), :↓]' - f[(1, 1), :↓]' * f[(1, 2), :↑]')
end
function triplet_0()
    1 / √2 * (f[(1, 1), :↑]' * f[(1, 2), :↓]' + f[(1, 1), :↓]' * f[(1, 2), :↑]')
end
function triplet_plus()
    1 / √2 * (f[(1, 1), :↑]' * f[(1, 2), :↑]' + f[(1, 1), :↓]' * f[(1, 2), :↓]')
end
function triplet_minus()
    1 / √2 * ((f[(1, 1), :↑]' * f[(1, 2), :↑]' - f[(1, 1), :↓]' * f[(1, 2), :↓]'))
end

function def_state(state_name, H)
    vac_ind = FermionicHilbertSpaces.state_index(FockNumber(0), H)
    H2, v0 = if ismissing(vac_ind)
        Haux = hilbert_space(keys(H), push!(copy(basisstates(H)), FockNumber(UInt(0))))
        Haux, vac_state(Haux)
    else
        H, vac_state(H)
    end
    v = matrix_representation(state_name(), H2; projection = true) * v0
    if ismissing(vac_ind)
        v = v[1:(end - 1)]
    end
    return normalize!(v)
end

max_mixed_state(H) = Matrix{ComplexF64}(I, dim(H), dim(H)) / dim(H)

function werner_state(state_name, p, H)
    (1 - p) * density_matrix(def_state(state_name, H)) + p * max_mixed_state(H)
end

random_state(H) = normalize!(randn(ComplexF64, dim(H)))
random_product_state(Hs, H) = generalized_kron(random_state.(Hs), Hs => H)
random_product_state(sys::QuantumDotSystem) = random_product_state(sys.Hs_main, sys.H_main)
function random_separable_state(nbr_states, Hs, H)
    p = rand(nbr_states)
    p = p ./ sum(p)
    ρ_sep = sum(p[i] * density_matrix(random_product_state(Hs, H)) for i in 1:nbr_states)
    return ρ_sep
end
function random_separable_state(N, sys::QuantumDotSystem)
    random_separable_state(N, sys.Hs_main, sys.H_main)
end
density_matrix(v::AbstractVector) = v * v'
density_matrix(ρ::AbstractMatrix) = ρ

function hilbert_schmidt_ensemble(H)
    d = dim(H)
    X = randn(ComplexF64, d, d)
    ρ = X'X / tr(X'X)
    return ρ
end

## =============== Ground States ====================
abstract type DiagonalizationAlg end
struct ExactDiagonalizationAlg <: DiagonalizationAlg end

function eig_state(m::AbstractMatrix, n, ::ExactDiagonalizationAlg)
    eigenvalues, eigenvectors = eigen(Matrix(m))
    eigenvectors[:, n]
end

using ArnoldiMethod
struct ArnoldiAlg <: DiagonalizationAlg end
function eig_state(m::AbstractMatrix, n, ::ArnoldiAlg; kwargs...)
    # https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl/issues/149
    decomp, history = try
        partialschur(Hermitian(m), nev = n, which = :SR; kwargs...)
    catch e
        @warn e "Trying to increase mindim and restarts"
        println(m)
        partialschur(Hermitian(m), nev = n, which = :SR; kwargs...,
            mindim = 40, maxdim = size(m, 1), restarts = 1000)
    end
    vals, vecs = partialeigen(decomp)
    idx = sortperm(real(vals))[n]
    return vecs[:, idx]
end

using KrylovKit
struct KrylovAlg <: DiagonalizationAlg end

function eig_state(m::AbstractMatrix, n, ::KrylovAlg; kwargs...)
    vals, vecs, info = eigsolve(
        m, n, :SR; kwargs...)
    if info.converged < n
        @warn "Only $(info.converged) eigenvalues converged, requested $n"
    end
    return vecs[n]
end

ground_state(m, alg = KrylovAlg()) = eig_state(m, 1, alg)
eig_state(m, n) = eig_state(m, n, KrylovAlg())
