# ViewFlux.jl

Viewflux is used to test custom made networks and visualize the training in order to quickly
optimize training parameters and network structure. Automatically uses gpu if able.

```julia
include(abspath("modules/Viewflux.jl"))
```

Uses the Flux.jl, Plots.jl, CUDA.jl, and ProgressMeter.jl libraries
