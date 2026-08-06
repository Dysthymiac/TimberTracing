# Timber Tracing

Self-contained experiment for matching sawn **boards** back to the **logs** they came
from, by training a `LogBoardAutoencoder` (U-Net encoders/decoders) to produce matching
"barcodes" for a log's heightmap and its boards, then matching by cosine similarity.

Reference implementation of the paper **"Timber tracing with multimodal encoder-decoder
networks"** (Zolotarev et al., CAIP 2019) — see [Citation](#citation).

This is a Julia package `TimberTracing` (code in [`src/`](src)); the experiment lives in the
top-level script [`TimberTracingTest.jl`](TimberTracingTest.jl), which does `using TimberTracing`.
Run everything with `julia --project=.`.

## Prerequisites

- **Julia 1.11**
- An NVIDIA GPU is optional — it runs on CPU otherwise (much slower). See **GPU notes** below.
- The dataset (not included). By default it is expected at
  `%TIMBER_TRACING_ROOT%/timber_tracing/<DATASET>/...`.

## Data location

Set the **`TIMBER_TRACING_ROOT`** environment variable to your data root — the folder that
contains a `timber_tracing/` sub-folder with the input data. Model checkpoints and result
images are written under `out_data/` in the same root. The experiment errors out with a
clear message if the variable is not set.

`TIMBER_TRACING_ROOT/` must contain a `timber_tracing/` folder holding your **dataset
folders** — see **Data format** below for the exact layout and file naming.

## Data format

Under `DATA_ROOT/timber_tracing/` you place one or more **dataset folders** — **any name**.
The folder name is the dataset id you refer to in the train/validation config; there is no
fixed list of allowed names (put your own data in a folder and use its name).

Inside a dataset folder (searched recursively), files are matched by name:

| Filename pattern | Meaning |
|---|---|
| `log<N>_clusters.<ext>`        | cluster image for log `N` |
| `log<N>_run<R>.<ext>`          | heightmap of log `N`, scan/run `R` (there can be several) |
| `log<N>_board<M>_<side>.<ext>` | board `M` of log `N`; `<side>` e.g. `upper` / `lower` |

`<N>`, `<R>`, `<M>` are integers; `<ext>` is any image format FileIO reads (png, tif, …).
`<side>` and the folder name must not contain `_` or `.`. Non-matching files are ignored; a
missing dataset or a folder with no matching files produces a **warning** (not a silent
empty run).

### Minimal example

```
$TIMBER_TRACING_ROOT/timber_tracing/
├── train/
│   ├── log1_clusters.png
│   ├── log1_run1.png
│   ├── log1_board1_upper.png
│   ├── log1_board1_lower.png
│   ├── log2_clusters.png
│   └── ...
└── val/
    ├── log1_clusters.png
    ├── log1_run1.png
    └── log1_board1_upper.png
```

### Choosing train / validation sets

`Args.train_proposal` / `Args.val_proposal` map dataset-folder names to a log selection
(`all` = every log in that folder):

```julia
train_proposal = Dict("train" => all)   # matches the example above
val_proposal   = Dict("val"   => all)
```

Edit them in `TimberTracingTest.jl`, or pass per call:
`train_model(1; train_proposal=Dict("train"=>all), val_proposal=Dict("val"=>all))`.
The shipped defaults reference the original paper's datasets (`HONKALAHTI2018`, `DIGISAW2021`,
`HONKALAHTI2019`) — change them to your own folder names.

## Running

```bash
# set your data root first (bash shown; use $env: on Windows PowerShell)
export TIMBER_TRACING_ROOT=/path/to/data
julia --project=. TimberTracingTest.jl
```

The first run instantiates the project (`julia --project=. -e 'using Pkg; Pkg.instantiate()'`
downloads/precompiles dependencies). Run as a script it runs the full experiment
(5 iterations × up to 50 epochs); in the REPL, `include("TimberTracingTest.jl")` runs a single
short iteration.

To experiment interactively, start `julia --project=.`, then `include("TimberTracingTest.jl")`
(this defines everything and runs one short iteration) and call
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

- Precompilation is forced **serial** (`JULIA_NUM_PRECOMPILE_TASKS=1`, set at the top of
  `TimberTracingTest.jl`) to avoid a Julia 1.11 deadlock when precompiling Flux's CUDA
  extensions in parallel. If you `instantiate` manually, set that env var first.
- On some Windows machines the bundled `CUDNN_jll` fails to load its sublibraries
  (`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`). If you hit this and have a working system
  cuDNN, point CUDA.jl's cuDNN at it via an `Overrides.toml` in your Julia depot and put the
  cuDNN `bin` on `PATH`. On Linux/clusters the default artifacts generally just work.

## Notes

- Built for **Flux 0.16** (explicit-optimiser API) and **PrettyTables 3.x**.
- Dependency versions are resolved by Julia (no manual `[compat]` pins); `Manifest.toml`
  records the exact working set.

## Citation

If you use this code, please cite the paper it implements:

> F. Zolotarev, T. Eerola, L. Lensu, H. Kälviäinen, H. Haario, J. Heikkinen, and T. Kauppi,
> "Timber tracing with multimodal encoder-decoder networks," in *International Conference on
> Computer Analysis of Images and Patterns (CAIP)*, Springer, 2019, pp. 342–353.
> doi:[10.1007/978-3-030-29891-3_30](https://doi.org/10.1007/978-3-030-29891-3_30)
> · [PDF](https://lutpub.lut.fi/bitstream/handle/10024/160199/zolotarev_et_al_timber_tracing_final_draft.pdf)

```bibtex
@inproceedings{zolotarev2019timber,
  title     = {Timber Tracing with Multimodal Encoder-Decoder Networks},
  author    = {Zolotarev, Fedor and Eerola, Tuomas and Lensu, Lasse and
               K{\"a}lvi{\"a}inen, Heikki and Haario, Heikki and Heikkinen, Jere and
               Kauppi, Tomi},
  booktitle = {International Conference on Computer Analysis of Images and Patterns (CAIP)},
  pages     = {342--353},
  year      = {2019},
  publisher = {Springer International Publishing},
  doi       = {10.1007/978-3-030-29891-3_30}
}
```
