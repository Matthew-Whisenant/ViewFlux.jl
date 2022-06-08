# * Simple Periodic ML Model

# Get network and plotting function
include(abspath("src/viewflux.jl"))

# ? Plotting options (num_epochs, epoch_plot, fps, gifname)
plotopts =
    let
        # Create plotting data
        num_epochs = 200
        epoch_plot = 50
        fps = 30
        gifname = "sinnet.gif"

        plotopts_struct(num_epochs, epoch_plot, fps, gifname)
    end

# ? Raw data options (x, y, use_cuda)
dataopts =
    let
        # Create input and output data
        actual(xᵢ) = sin.(xᵢ) .+ 4
        x = [Float32(i) for i ∈ range(0, 4 * π, 100)]
        y = [actual(i) for i in x]
        use_cuda = true

        dataopts_struct(x, y, use_cuda)
    end

# ? Network options (model, opt, batch_size, shuf_data, ratio, seq_length)
mlopts =
    let in = 1, out = 1
        models =
            [
                let nodes = 5
                    Chain(
                        Dense(in => nodes, sin),
                        Dense(nodes => nodes),
                        Dense(nodes => out)
                    )
                end,
                let nodes = 5
                    Chain(
                        Dense(in => nodes, sin),
                        Dense(nodes => nodes),
                        Dense(nodes => out)
                    )
                end
            ]
        num_nets = length(models)
        optimisers = fill(Optimiser(ADAM(1.0f-2)), num_nets)
        batch_size = fill(1, num_nets)
        shuf_data = fill(false, num_nets)
        ratio = fill(0.8, num_nets)
        seq_length = [1, 2]

        mlopts_struct(models, optimisers, batch_size, shuf_data, ratio, seq_length)
    end

# Training and visualize
mlrun(mlopts, dataopts, plotopts)

return nothing