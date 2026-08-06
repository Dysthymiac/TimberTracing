# Timber Tracing

Self-contained experiment for matching sawn **boards** back to the **logs** they came
from, by training a `LogBoardAutoencoder` (U-Net encoders/decoders) to produce matching
"barcodes" for a log's heightmap and its boards, then matching by cosine similarity.

Entry point: [`TimberTracingTest.jl`](TimberTracingTest.jl). Local modules live in
[`src/`](src) and are loaded via `Config.jl` (they are not a registered package).

## Prerequisites

- **Julia 1.11**
- An NVIDIA GPU is optional — it runs on CPU otherwise (much slower). See **GPU notes** below.
- The dataset (not included). By default it is expected at
  `%TIMBER_TRACING_ROOT%/timber_tracing/<DATASET>/...`.

## Data location

`Config.jl` defines where data lives:

```julia
const DATA_ROOT = get(ENV, "TIMBER_TRACING_ROOT", raw"D:\OneDrive\OneDrive - LUT University")
dataset_path() = joinpath(DATA_ROOT, "timber_tracing")   # input datasets
out_path(parts...) = joinpath(DATA_ROOT, "out_data", parts...)   # checkpoints + result images
```

Point it at your data root by setting the `TIMBER_TRACING_ROOT` environment variable
(or editing the default). The dataset root must contain a `timber_tracing/` folder whose
subfolders are dataset names (`HONKALAHTI2018`, `HONKALAHTI2019`, `DIGISAW2021`,
`HONKALAHTIOLD`, ...).

## Running

```bash
julia --project=. TimberTracingTest.jl
```

The first run instantiates the project (downloads/precompiles dependencies). In the REPL,
`include("TimberTracingTest.jl")` runs a single short iteration; run as a script it runs the
full experiment (5 iterations × 50 epochs).

To experiment interactively, `include("Config.jl")` then call
`train_model(1; epochs=1, batchsize=1, target_size=(80,256))` with whatever overrides you want.

### Following a long run

Run it in a terminal to watch progress live. Julia **block-buffers stdout when it is
redirected to a file**, so `julia ... > run.log` will look frozen for a long time. To get a
saved log that updates live, run through a driver that flushes periodically:

```julia
# follow.jl
@async while true; try; flush(stdout); flush(stderr); catch; end; sleep(1); end
include("TimberTracingTest.jl")
```

```bash
julia --project=. follow.jl > run.log 2>&1   # then: tail -f run.log
```

## GPU notes

- Precompilation is forced **serial** (`JULIA_NUM_PRECOMPILE_TASKS=1`, set in `Config.jl`)
  to avoid a Julia 1.11 deadlock when precompiling Flux's CUDA extensions in parallel.
- On some Windows machines the bundled `CUDNN_jll` fails to load its sublibraries
  (`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`). If you hit this and have a working system
  cuDNN, point CUDA.jl's cuDNN at it via an `Overrides.toml` in your Julia depot and put the
  cuDNN `bin` on `PATH`. On Linux/clusters the default artifacts generally just work.

## Notes

- Built for **Flux 0.16** (explicit-optimiser API) and **PrettyTables 3.x**.
- Dependency versions are resolved by Julia (no manual `[compat]` pins); `Manifest.toml`
  records the exact working set.
