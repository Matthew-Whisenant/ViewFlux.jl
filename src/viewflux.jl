# Standard imports
using Flux, GLMakie, CUDA, ProgressMeter

# Specific imports
using Flux: DataLoader, Optimiser
using Flux: update!, params, reset!, mse

# Disallow slow scalar gpu indexing
has_cuda() && CUDA.allowscalar(false)

# Structure definitions
include("structs.jl")

# Function definitions
include("mldata.jl")
include("mldevice.jl")
include("mlrun.jl")
include("mldebug.jl")
include("mlplots.jl")
include("mltrain.jl")

return nothing