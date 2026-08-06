module TimberTracing

using Reexport

# Local support modules. Order matters: ClassificationMetrics uses CommonUtils.
include("CommonUtils.jl")
include("ClassificationMetrics.jl")
include("ExtraAugmentors.jl")
include("TimberDatasets.jl")
include("TimberTraining.jl")

# Bring each submodule's exported names into scope for `using TimberTracing`.
@reexport using .CommonUtils
@reexport using .ClassificationMetrics
@reexport using .ExtraAugmentors
@reexport using .TimberDatasets
@reexport using .TimberTraining

# Also expose the submodules themselves, for qualified names
# (e.g. TimberDatasets.board, TimberTraining.equivariant_model_loss).
export CommonUtils, ClassificationMetrics, ExtraAugmentors, TimberDatasets, TimberTraining

end # module
