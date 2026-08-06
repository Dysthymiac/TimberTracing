# Julia 1.11 can deadlock when precompiling Flux's CUDA extensions in parallel
# (ConcurrencyViolationError: FluxCUDAExt -> FluxCUDAcuDNNExt). Precompile serially.
# Override by setting JULIA_NUM_PRECOMPILE_TASKS yourself before launching Julia.
get!(ENV, "JULIA_NUM_PRECOMPILE_TASKS", "1")

isinteractive() && using Revise

# Activate the isolated Timber Tracing project and make sure its deps are installed.
if !startswith(Base.active_project(), @__DIR__)
    using Pkg
    println("Activating project at $(@__DIR__)...")
    Pkg.activate(@__DIR__)
    Pkg.instantiate(; verbose=true)
    println("Project activated")
end

# Make the local Timber Tracing modules in src/ available to `using`.
if joinpath(@__DIR__, "src/") ∉ LOAD_PATH
    push!(LOAD_PATH, joinpath(@__DIR__, "src/"))
    println("Added local modules")
end

# ---- Data locations -------------------------------------------------------
# Root directory that holds the `timber_tracing` dataset. Model checkpoints and
# result images are written under `out_data/` inside the same root.
# Override on other machines by setting the TIMBER_TRACING_ROOT env var.
const DATA_ROOT = get(ENV, "TIMBER_TRACING_ROOT", raw"D:\OneDrive\OneDrive - LUT University")

dataset_path() = joinpath(DATA_ROOT, "timber_tracing")
out_path(parts...) = joinpath(DATA_ROOT, "out_data", parts...)
