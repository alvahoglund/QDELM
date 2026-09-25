# =============================================================================
# bench_propagators.jl — speed + accuracy of every propagation path on YOUR problem.
#
#   julia --project -t auto benchmark/bench_propagators.jl        # runs the examples at the bottom
#
# Interactive:
#   include("benchmark/bench_propagators.jl")
#   st = BenchSettings(atol = 1e-8, mode = :both)
#   bench_system(; nbr_dots_res = 3, qn_res = 2, ham_param_funcs = hpf, ts = [5.0], settings = st)
#   bench_system(; nbr_dots_res = 3, qn_res = 2, ham_param_funcs = hpf, ts = 0.5:0.5:5.0, settings = st)
#   bench_scan([(; nbr_dots_res = 2, qn_res = 1), (; nbr_dots_res = 3, qn_res = 2)];
#              ham_param_funcs = hpf, ts = [5.0], settings = st)
#   bench_synthetic(; N = 4000, p = 16, ρt = 100, nt = 1, settings = st)
#   calibrate()          # machine constants for AutoPropagatorAlg
# =============================================================================
using LinearAlgebra, SparseArrays, Random, Printf, Statistics
using QDELM
using ExponentialUtilities
const Q = QDELM

# ─────────────────────────────── settings ────────────────────────────────────
Base.@kwdef struct BenchSettings
    tol::Float64 = 1e-10          # tolerance handed to every method that has one
    atol::Float64 = 1e-8          # accuracy you NEED (max column 2-norm error of U) → ✓ / ranking
    algs::Any = :default          # :fast | :default | :tuning | Vector{Pair{String,Any}}
    samples::Int = 5              # timed repetitions
    max_seconds::Float64 = 20.0   # per-method budget; slower methods are timed once
    mode::Symbol = :latency       # :latency | :throughput (nthreads concurrent calls) | :both
    blas_threads::Int = 1         # 1 matches the threaded ensemble loop
    diag_max::Int = 1000          # skip dense diagonalization above this N
    expprop_max::Int = 500       # skip dense exp (QPExpProp) above this N
    skip::Vector{String} = String[]
    csv::Union{Nothing, String} = nothing   # append results to this CSV file
end

# ───────────────────────────── algorithm sets ────────────────────────────────
function algorithm_set(set, tol)
    set isa AbstractVector && return set
    fast = Pair{String, Any}[
        "diag"      => Q.DiagonalizationPropagatorAlg(),
        "cheb"      => Q.ChebyshevPropagatorAlg(; tol),
        "auto"      => Q.AutoPropagatorAlg(; tol),
    ]
    set === :fast && return fast
    default = vcat(fast, Pair{String, Any}[
        "cheb-gersh"      => Q.ChebyshevPropagatorAlg(; tol, bounds = :gershgorin),
        "eu-lanczos"      => Q.ExpUtilsLanczosPropagatorAlg(; tol),
        "eu-timestep"     => Q.ExpUtilsTimestepPropagatorAlg(; tol),
    ])
    set === :default && return default
    set === :tuning && return vcat(default, Pair{String, Any}[
        "cheb-margin5%"   => Q.ChebyshevPropagatorAlg(; tol, margin = 0.05),
        "cheb-onepass"    => Q.ChebyshevPropagatorAlg(; tol, multitime = :onepass),
        "cheb-stepping"   => Q.ChebyshevPropagatorAlg(; tol, multitime = :stepping),
        "eu-lanczos-m100"  => Q.ExpUtilsLanczosPropagatorAlg(; tol, m = 100),
        "eu-lanczos-m60"  => Q.ExpUtilsLanczosPropagatorAlg(; tol, m = 60),
    ])
    throw(ArgumentError("unknown algorithm set $set"))
end

function skip_reason(name, alg, N, st, slow)
    name in st.skip && return "skipped: user"
    name in slow && return "skipped: too slow at a smaller size"
    alg isa Q.DiagonalizationPropagatorAlg && N > st.diag_max && return "skipped: N > diag_max"
    alg isa Q.QPAlg{Q.QPExpProp} && N > st.expprop_max && return "skipped: N > expprop_max"
    return nothing
end

# ─────────────────────────────── helpers ─────────────────────────────────────
function random_hamiltonian(N; nnz_per_row = 12, real_ham = false, seed = 1)
    rng = Xoshiro(seed)
    T = real_ham ? Float64 : ComplexF64
    A = sprand(rng, T, N, N, nnz_per_row / (2N))
    nonzeros(A) .-= real_ham ? 0.5 : 0.5 + 0.5im
    return A + A' + spdiagm(0 => randn(rng, N))
end
random_isometry(N, p; seed = 2) = Matrix(qr(randn(Xoshiro(seed), ComplexF64, N, p)).Q)[:, 1:p]
maxcolerr(Us, Vs) = maximum(maximum(norm, eachcol(U - V)) for (U, V) in zip(Us, Vs))
fmt(x) = isnan(x) ? "—" : @sprintf("%.1e", x)
ftime(x) = isnan(x) ? "—" : @sprintf("%.4f", x)

function warmup(algs, H)
    Hs = random_hamiltonian(64; seed = 7, real_ham = eltype(H) <: Real)
    Ψs = random_isometry(64, 3; seed = 8)
    for (_, alg) in algs
        try
            Q.propagate_block(Hs, Ψs, [0.3, 0.6], alg)
        catch
        end
    end
end

function time_method(f, st)
    s = @timed f()
    out = s.value
    times = [s.time]
    bytes = s.bytes
    spent = s.time
    while length(times) < st.samples && spent + minimum(times) ≤ st.max_seconds
        s = @timed f()
        push!(times, s.time)
        spent += s.time
        bytes = min(bytes, s.bytes)
    end
    return out, minimum(times), median(times), bytes, length(times)
end

"Wall time per call when nthreads() calls run concurrently (what the ensemble loop sees)."
function time_throughput(f)
    n = Threads.nthreads()
    t = @elapsed foreach(wait, [Threads.@spawn(f()) for _ in 1:n])
    return t / n
end

# ─────────────────────────── problem & reference ─────────────────────────────
function describe_problem(H, Ψ0, ts, st)
    Hop = Q._operator(H)
    N, p = size(Ψ0)
    lo, hi = Q.lanczos_bounds(Hop)
    ρ = (hi - lo) / 2
    tsv = Q._tvec(ts)
    τ = sort(abs.(tsv))
    tmax = τ[end]
    Δ = length(τ) > 1 ? maximum(diff([0.0; τ])) : tmax
    K = length(first(Q.chebyshev_coefficients(1.01 * ρ * tmax, st.tol))) - 1
    prob = (; N, p, nnz_per_row = Q._nnz_per_row(Hop), λmin = lo, λmax = hi, ρ, tmax,
        nt = length(tsv), ρt = ρ * tmax, ρΔt = ρ * Δ, cheb_terms = K, dense_GB = 3 * 16 * N^2 / 1e9)
    @printf("N = %d   N_main (p) = %d   nnz/row = %.1f   eltype(H) = %s\n", N, p, prob.nnz_per_row, eltype(H))
    @printf("spectrum ≈ [%.4g, %.4g]   ρ (half-width) = %.4g\n", lo, hi, ρ)
    @printf("t_max = %.4g   n_t = %d   ρ·t_max = %.1f   ρ·Δt = %.1f\n", tmax, prob.nt, prob.ρt, prob.ρΔt)
    @printf("Chebyshev terms for t_max at tol=%.0e: %d   dense diag memory ≈ %.2f GB   threads: julia %d / BLAS %d\n",
        st.tol, K, prob.dense_GB, Threads.nthreads(), BLAS.get_num_threads())
    return prob
end

function reference_solution(H, Ψ0, ts, st)
    tight = Q.ChebyshevPropagatorAlg(; tol = 1e-13, bounds = :gershgorin)
    if size(H, 1) ≤ st.diag_max
        ref, _ = Q.propagate_block(H, Ψ0, ts, Q.DiagonalizationPropagatorAlg())
        alt, _ = Q.propagate_block(H, Ψ0, ts, tight)
        return ref, maxcolerr(ref, alt), "diagonalization (cross-checked vs Chebyshev/Gershgorin 1e-13)"
    else
        ref, _ = Q.propagate_block(H, Ψ0, ts, tight)
        alt, _ = Q.propagate_block(H, Ψ0, ts, Q.ExpUtilsLanczosPropagatorAlg(; tol = 1e-12))
        return ref, maxcolerr(ref, alt), "Chebyshev/Gershgorin 1e-13 (cross-checked vs eu-lanczos 1e-12)"
    end
end

# ─────────────────────────────── running ─────────────────────────────────────
const EMPTY_ROW = (; chosen = "", tmin = NaN, tmed = NaN, tput = NaN, MB = NaN, nrep = 0,
    err_U = NaN, err_S = NaN, err_est = NaN, iso = NaN, nmatvec = -1)

function run_one(name, alg, H, Ψ0, ts, ref, ref_rows, measurements, st)
    f = () -> Q.propagate_block(H, Ψ0, ts, alg)
    try
        (Us, info), tmin, tmed, bytes, nrep = time_method(f, st)
        tput = (st.mode in (:throughput, :both) && tmin ≤ st.max_seconds) ? time_throughput(f) : NaN
        err_U = maxcolerr(Us, ref)
        err_S = ref_rows === nothing ? NaN :
                maximum(maximum(abs, Q.measurement_rows(U, measurements) .- R) for (U, R) in zip(Us, ref_rows))
        iso = maximum(opnorm(U' * U - I) for U in Us)
        return (; name, status = "ok", chosen = String(info.method), tmin, tmed, tput,
            MB = bytes / 2^20, nrep, err_U, err_S, err_est = Float64(info.err_est), iso,
            nmatvec = Int(info.nmatvec))
    catch e
        e isa InterruptException && rethrow()
        return merge((; name, status = "ERROR: " * first(sprint(showerror, e), 70)), EMPTY_ROW)
    end
end

function print_header()
    @printf("%-18s %-24s %10s %10s %10s %9s %9s %9s %9s %9s %7s  %s\n", "method", "chosen", "t_min[s]",
        "t_med[s]", "t/samp[s]", "alloc MB", "err_U", "err_S", "err_est", "|U'U-I|", "matvec", "status")
end

function print_row(r, st)
    mark = r.status == "ok" ? (r.err_U ≤ st.atol ? "✓" : "✗ inaccurate") : r.status
    @printf("%-18s %-24s %10s %10s %10s %9s %9s %9s %9s %9s %7s  %s\n", r.name, r.chosen, ftime(r.tmin),
        ftime(r.tmed), ftime(r.tput), isnan(r.MB) ? "—" : @sprintf("%.1f", r.MB), fmt(r.err_U),
        fmt(r.err_S), fmt(r.err_est), fmt(r.iso), r.nmatvec < 0 ? "—" : string(r.nmatvec), mark)
end

function ranking(results, atol)
    ok = filter(r -> r.status == "ok" && r.err_U ≤ atol, results)
    return sort(ok; by = r -> r.tmin)
end

function print_summary(results, ref_unc, st)
    ok = ranking(results, st.atol)
    println("\n── ranking: correct methods (err_U ≤ $(st.atol)), fastest first ──")
    if isempty(ok)
        println("  no method reached atol — loosen atol or tighten tol")
        return
    end
    for (i, r) in enumerate(ok)
        @printf("  %2d. %-18s %10.4f s  (%5.2f× best)   err_U = %.1e\n", i, r.name, r.tmin, r.tmin / ok[1].tmin, r.err_U)
    end
    ia = findfirst(r -> r.name == "auto", results)
    if ia !== nothing && results[ia].status == "ok"
        a = results[ia]
        @printf("  auto chose %s: %.4f s (%.2f× best)%s\n", a.chosen, a.tmin, a.tmin / ok[1].tmin,
            a.tmin > 1.3 * ok[1].tmin ? "   → run calibrate() / adjust small_dim, max_diag_dim" : "")
    end
    if st.mode in (:throughput, :both)
        okt = sort(filter(r -> !isnan(r.tput), ok); by = r -> r.tput)
        isempty(okt) || @printf("  best throughput (%d threads): %s at %.4f s/sample\n",
            Threads.nthreads(), okt[1].name, okt[1].tput)
    end
    ref_unc > st.atol / 10 &&
        println("  ⚠ reference uncertainty $(fmt(ref_unc)) is not ≪ atol — accuracy numbers unreliable")
end

function save_csv(path, results, prob)
    pcols = (:N, :p, :nnz_per_row, :ρ, :tmax, :nt, :ρt, :ρΔt)
    rcols = (:name, :status, :chosen, :tmin, :tmed, :tput, :MB, :nrep, :err_U, :err_S, :err_est, :iso, :nmatvec)
    newfile = !isfile(path)
    open(path, "a") do io
        newfile && println(io, join(string.((pcols..., rcols...)), ","))
        for r in results
            vals = ([getproperty(prob, c) for c in pcols]..., [getproperty(r, c) for c in rcols]...)
            println(io, join((replace(string(v), "," => ";") for v in vals), ","))
        end
    end
end

"""
    benchmark_problem(H, Ψ0, ts; measurements = nothing, settings, slow, title)
Times every algorithm on exp(-iH t) Ψ0 and measures its error against a cross-checked
reference. `measurements` (diagonal operators) adds err_S = max |Δ scrambling-map entry|.
"""
function benchmark_problem(H, Ψ0, ts; measurements = nothing, settings = BenchSettings(),
        slow = Set{String}(), title = "")
    st = settings
    BLAS.set_num_threads(st.blas_threads)
    algs = algorithm_set(st.algs, st.tol)
    warmup(algs, H)
    println("\n", "="^120, "\n", title, "\n", "-"^120)
    prob = describe_problem(H, Ψ0, ts, st)
    ref, ref_unc, refname = reference_solution(H, Ψ0, ts, st)
    ref_rows = measurements === nothing ? nothing : [Q.measurement_rows(U, measurements) for U in ref]
    @printf("reference: %s   uncertainty ≈ %s\n\n", refname, fmt(ref_unc))
    print_header()
    results = NamedTuple[]
    for (name, alg) in algs
        reason = skip_reason(name, alg, size(H, 1), st, slow)
        r = reason === nothing ? run_one(name, alg, H, Ψ0, ts, ref, ref_rows, measurements, st) :
            merge((; name, status = reason), EMPTY_ROW)
        push!(results, r)
        print_row(r, st)
        (r.status == "ok" && r.tmin > st.max_seconds) && push!(slow, name)
    end
    print_summary(results, ref_unc, st)
    st.csv === nothing || save_csv(st.csv, results, prob)
    return (; problem = prob, ref_uncertainty = ref_unc, results)
end

# ─────────────────────────── user-facing entry points ────────────────────────
"Build one Hamiltonian sample exactly as get_ensemble_average does (edit if names differ)."
function build_problem(; nbr_dots_res, qn_res, ham_param_funcs, seed = 1)
    Random.seed!(seed)
    grid = QDELM.generate_grid(2, nbr_dots_res)
    sys = tight_binding_system(grid, qn_res)
    measurements = QDELM.charge_probabilities(sys)
    ham_symb = QDELM.hamiltonians(grid, ham_param_funcs)
    hams_mat = QDELM.matrix_representation_hams(ham_symb, sys)
    ψres = ground_state(hams_mat.res)
    Ψ0 = Q.initial_block(sys.H_main, sys.H_res, sys.H_total, ψres)
    return (; H = hams_mat.total, Ψ0, measurements, sys, ψres)
end

"""
    bench_system(; nbr_dots_res, qn_res, ham_param_funcs, ts = nothing, t_func = nothing,
                   nhams = 1, seed = 1, settings = BenchSettings())
Benchmark on your actual system. Give `ts` (number, vector, or range) or `t_func`
(called once per Hamiltonian sample, as in get_ensemble_average).
"""
function bench_system(; nbr_dots_res, qn_res, ham_param_funcs, ts = nothing, t_func = nothing,
        nhams = 1, seed = 1, settings = BenchSettings(), slow = Set{String}())
    (ts === nothing && t_func === nothing) && throw(ArgumentError("give ts or t_func"))
    out = map(1:nhams) do h
        prob = build_problem(; nbr_dots_res, qn_res, ham_param_funcs, seed = seed + h - 1)
        tt = ts === nothing ? t_func() : ts
        benchmark_problem(prob.H, prob.Ψ0, tt; measurements = prob.measurements, settings, slow,
            title = "system: nbr_dots_res=$nbr_dots_res qn_res=$qn_res | Hamiltonian sample $h | ts=$(tt)")
    end
    return nhams == 1 ? only(out) : out
end

"""
    bench_synthetic(; N, p = 16, ρt = 100, nt = 1, nnz_per_row = 12, real_ham = false, settings)
Random sparse Hermitian H with the given size and dimensionless time ρ·t_max
(nt > 1: uniform grid t_max/nt : t_max).
"""
function bench_synthetic(; N, p = 16, ρt = 100.0, nt = 1, nnz_per_row = 12, real_ham = false,
        settings = BenchSettings(), slow = Set{String}())
    H = random_hamiltonian(N; nnz_per_row, real_ham)
    Ψ0 = random_isometry(N, p)
    lo, hi = Q.lanczos_bounds(H)
    tmax = ρt / ((hi - lo) / 2)
    ts = nt == 1 ? [tmax] : collect(range(tmax / nt, tmax; length = nt))
    benchmark_problem(H, Ψ0, ts; settings, slow,
        title = "synthetic: N=$N p=$p ρ·t_max=$ρt n_t=$nt nnz/row≈$nnz_per_row")
end

"""
    bench_scan(configs; ham_param_funcs, ts = nothing, settings)
`configs`: vector of NamedTuples with `nbr_dots_res`, `qn_res` (optionally `ts`), ordered
small → large. Methods exceeding max_seconds are skipped for later configs.
Prints a crossover table: which method wins where.
"""
function bench_scan(configs; ham_param_funcs, ts = nothing, settings = BenchSettings())
    slow = Set{String}()
    runs = map(configs) do c
        tt = haskey(c, :ts) ? c.ts : ts
        c => bench_system(; nbr_dots_res = c.nbr_dots_res, qn_res = c.qn_res, ham_param_funcs,
            ts = tt, settings, slow)
    end
    println("\n", "="^120, "\nSCAN SUMMARY (atol = $(settings.atol))")
    @printf("%-32s %7s %8s %5s   %-18s %10s   %-18s %10s   %s\n", "config", "N", "ρ·t", "n_t",
        "best", "t[s]", "runner-up", "t[s]", "auto")
    for (c, r) in runs
        rk = ranking(r.results, settings.atol)
        b1 = length(rk) ≥ 1 ? rk[1] : nothing
        b2 = length(rk) ≥ 2 ? rk[2] : nothing
        ia = findfirst(x -> x.name == "auto", r.results)
        a = ia === nothing ? nothing : r.results[ia]
        @printf("%-32s %7d %8.1f %5d   %-18s %10s   %-18s %10s   %s\n",
            "dots=$(c.nbr_dots_res) qn=$(c.qn_res)", r.problem.N, r.problem.ρt, r.problem.nt,
            b1 === nothing ? "—" : b1.name, b1 === nothing ? "—" : ftime(b1.tmin),
            b2 === nothing ? "—" : b2.name, b2 === nothing ? "—" : ftime(b2.tmin),
            (a === nothing || a.status != "ok") ? "—" : "$(a.chosen) $(ftime(a.tmin)) s")
    end
    return runs
end

"""
    calibrate(; N_eig = 1500, N_sp = 20_000, p = 16)
Machine constants for AutoPropagatorAlg's cost model (single BLAS thread).
"""
function calibrate(; N_eig = 1500, N_sp = 20_000, p = 16, nnz_per_row = 12)
    BLAS.set_num_threads(1)
    tmin(f) = (f(); minimum(@elapsed(f()) for _ in 1:3))
    A = randn(ComplexF64, N_eig, N_eig); A = Hermitian(A + A')
    Ar = randn(N_eig, N_eig); Ar = Hermitian(Ar + Ar')
    t_eig = tmin(() -> eigen(A))
    t_eigr = tmin(() -> eigen(Ar))
    V = Matrix(eigen(A).vectors); C = randn(ComplexF64, N_eig, p); Y = similar(C)
    t_gemm = tmin(() -> mul!(Y, V, C))
    H = random_hamiltonian(N_sp; nnz_per_row)
    X = randn(ComplexF64, N_sp, p); Z = similar(X)
    t_spmm = tmin(() -> mul!(Z, H, X, 2.0, -1.0))
    t_vec = tmin(() -> axpy!(0.3 + 0.1im, X, Z))
    c = (; sec_eig = t_eig / N_eig^3, sec_gemm = t_gemm / (N_eig^2 * p),
        sec_spmm = t_spmm / (nnz(H) * p), sec_vec = t_vec / (N_sp * p), real_eig_factor = t_eigr / t_eig)
    @printf("AutoPropagatorAlg(sec_eig = %.3e, sec_gemm = %.3e, sec_spmm = %.3e, sec_vec = %.3e, real_eig_factor = %.2f)\n",
        c.sec_eig, c.sec_gemm, c.sec_spmm, c.sec_vec, c.real_eig_factor)
    return c
end

# ───────────────────────────────── examples ──────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    st = BenchSettings(; atol = 1e-8, mode = :both, csv = "bench_results.csv")
    calibrate()
    bench_synthetic(; N = 500, p = 16, ρt = 50, settings = st)
    bench_synthetic(; N = 1000, p = 16, ρt = 50, settings = st)
    bench_synthetic(; N = 2000, p = 16, ρt = 50, settings = st)
    bench_synthetic(; N = 8000, p = 16, ρt = 50, settings = st)
    # bench_synthetic(; N = 8000, p = 16, ρt = 50, nt = 10, settings = st)
    # ── your system (edit) ──
    # include("my_ham_params.jl")            # defines ham_param_funcs, t_func
    # ham_param_funcs = QDELM.random_param_functions()
    # bench_system(; nbr_dots_res = 3, qn_res = 2, ham_param_funcs, [10], nhams = 2, settings = st)
    # bench_scan([(; nbr_dots_res = 2, qn_res = 1), (; nbr_dots_res = 3, qn_res = 2)];
    #            ham_param_funcs, ts = [5.0], settings = st)
end
