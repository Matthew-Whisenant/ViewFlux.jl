# * Simple Linear ML Model

# Get network and plotting function
include(abspath("src/viewflux.jl"))

# ? Plotting options (num_epochs, epoch_plot, fps, gifname)
plotopts = plotopts_struct(50, 5, 30, "simplenet.gif")

# ? Raw data options (x, y, use_cuda)
dataopts =
    let
        # Create input and output data
        actual(x) = 4 .* x .+ 2
        x = [Float32(i) for i = 0:10]
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
                let nodes = 10
                    Chain(
                        Dense(in => nodes, sin),
                        Dense(nodes => nodes),
                        Dense(nodes => out)
                    )
                end
            ]
        num_nets = length(models)
        optimisers = fill(Optimiser(Descent(1.0f-2)), num_nets)
        batch_size = fill(1, num_nets)
        shuf_data = fill(false, num_nets)
        ratio = fill(0.8, num_nets)
        seq_length = fill(1, num_nets)

        mlopts_struct(models, optimisers, batch_size, shuf_data, ratio, seq_length)
    end

# Training and visualize
#mlrun(mlopts, dataopts, plotopts)

return nothing