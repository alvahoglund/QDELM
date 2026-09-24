using TestItemRunner
@run_package_tests verbose = true

include("test_hamiltonian.jl")
include("test_measurements.jl")
include("test_states.jl")
include("test_time_evolution.jl")
include("test_scramblingmap.jl")