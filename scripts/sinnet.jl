# * Simple Linear ML Model

# Get network and plotting function
include(abspath("src/viewflux.jl"))

# Plotting options (num_epochs, epoch_plot, fps, gifname)
plotopts = plotopts_struct(150, 10, 30, "sinnet.gif")

# Training data setup
actual(x) = sin.(x) .+ 4
x = [[Float32(i)] for i ∈ range(0, 4 * π, 100)]
y = [actual(i) for i in x]

# Training data options (x, y, use_cuda)
dataopts = dataopts_struct(x, y, true)

# Network Structures
mlmodel1 =
    let nodes = 5, in = 1, out = 1
        Chain(
            Dense(in => nodes, sin),
            Dense(nodes => nodes),
            Dense(nodes => out)
        )
    end

mlmodel2 =
    let nodes = 5, in = 1, out = 1
        Chain(
            Dense(in => nodes, sin),
            Dense(nodes => nodes),
            Dense(nodes => out)
        )
    end

# Network optimiser
opt = Flux.Optimiser(ADAM(1.0f-2))

# Network options (mlmodel, opt, batch_size, shuf_data, ratio, seq_length)
mlopts1 = mlopts_struct(mlmodel1, opt, 1, false, 0.8, 1)
mlopts2 = mlopts_struct(mlmodel2, opt, 1, false, 0.8, 2)
multimlopts = [mlopts1, mlopts2]

# Training and visualize
mltrain(multimlopts, dataopts, plotopts)

return nothing