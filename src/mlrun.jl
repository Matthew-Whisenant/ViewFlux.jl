"""

    mlrun(mlopts, dataopts, plotopts)

        mlopts   | Machine learning options. See struct mlopts.
        dataopts | Raw data. See struct dataopts.
        plotopts | Plotting options. See struct plotopts.

"""
function mlrun(mlopts, dataopts, plotopts)

    # Load models to selected device (CPU or GPU)
    mydevice = mldevice(mlopts, dataopts)

    # Split and assemble data, place on gpu or cpu
    netdata = mldata(mlopts, dataopts, mydevice)

    # Get network parameters
    parameters = params.(mlopts.model)

    # Loss function
    loss(x, y, model) = sum(mse(model(xi)[:, end], yi) for (xi, yi) in zip(x, y)) / length(x)

    # Set up first frame of gif
    fig, yp_model, sl = mlplot(mlopts, dataopts, netdata)
    display(fig)

    # Now do training repeatedly
    @showprogress 0.1 "Training..." for epoch in 1:plotopts.num_epochs

        # Update epoch count
        push!(epoch_vec[], epoch)

        # For each network
        for i = 1:num_nets

            reset!(mlopts.model[i])

            # For each element in train_data
            for d in netdata.train[i]

                # Get network gradients
                grads = gradient(parameters[i]) do
                    loss(d..., mlopts.model[i])
                end

                # Update network parameters
                update!(mlopts.opt[i], parameters[i], grads)

            end

            # Append network losses
            reset!(mlopts.model[i])
            loss_vec_train[i].val = push!(loss_vec_train[i][], loss(netdata.train[i].data..., mlopts.model[i]))
            loss_vec_test[i].val = push!(loss_vec_test[i][], loss(netdata.test[i].data..., mlopts.model[i]))

            # Append predicted model output
            reset!(mlopts.model[i])
            yp_train = reduce(vcat,
                [mlopts.model[i](xi)[:, end] for
                 xi in netdata.train[i].data[1]]) |> cpu
            yp_test = reduce(vcat,
                [mlopts[i].model[i](xi)[:, end] for
                 xi in netdata.test[i].data[1]]) |> cpu
            yp_model.val = push!(yp_model[], [yp_train; yp_test])

            # Save next frame of gif
            if mod(epoch, plotopts.epoch_plot) == 0

                sl.range = 0:epoch_vec[][end]
                autolimits!.(fig.content[1:2])
                sl.selected_index = epoch - 1
                sleep(0.001)

            end

        end

    end

    #display(gif(anim, joinpath("temp", plotopts.gifname), fps=plotopts.fps))

    return nothing

end