"""

    mltrain(multimlopts, dataopts, plotopts)

    multimlopts | Vector of mlopts. See struct mlopts.
    dataopts    | Training and testing data. See struct dataopts.
    plotopts    | Plotting options. See struct plotopts.

    Training function. Automatically does plotting according to
    variables set in plotopts.

"""
function mltrain(multimlopts, dataopts, plotopts)

    # Get device to use
    if dataopts.use_cuda && CUDA.functional()
        device = gpu
    else
        device = cpu
    end

    # Get number of networks
    num_nets = length(multimlopts)

    # Set models on device
    [multimlopts[i].mlmodel = multimlopts[i].mlmodel |> device for i = 1:num_nets]

    # Get network parameters
    parameters = [Flux.params(multimlopts[i].mlmodel) for i = 1:num_nets]

    # Loss function
    function loss(x, y, model)

        #Flux.reset!(m)

        sum(mse(model(xi)[:, end], yi) for (xi, yi) in zip(x, y)) / length(x)

    end

    # Split and assemble data, place on gpu or cpu
    netdata = [mldata(multimlopts[i], dataopts, device) for i = 1:num_nets]

    # Initial epoch network state
    epoch_vec = Observable([0])
    loss_vec_train = [Observable([loss(netdata[i].train.data..., multimlopts[i].mlmodel)]) for i = 1:num_nets]
    loss_vec_test = [Observable([loss(netdata[i].test.data..., multimlopts[i].mlmodel)]) for i = 1:num_nets]

    # Set up first frame of gif
    fig, yp_model, sl = mlplot(multimlopts, dataopts, netdata, epoch_vec, loss_vec_train, loss_vec_test)
    display(fig)

    # Now do training repeatedly
    @showprogress 0.1 "Training..." for epoch in 1:plotopts.num_epochs

        # Update epoch count
        epoch_vec.val = push!(epoch_vec[], epoch)

        # For each network
        for i = 1:num_nets

            reset!(multimlopts[i].mlmodel)

            # For each element in train_data
            for d in netdata[i].train

                # Get network gradients
                grads = gradient(parameters[i]) do
                    loss(d..., multimlopts[i].mlmodel)
                end

                # Update network parameters
                update!(multimlopts[i].opt, parameters[i], grads)

            end

            # Append network losses
            reset!(multimlopts[i].mlmodel)
            loss_vec_train[i].val = push!(loss_vec_train[i][], loss(netdata[i].train.data..., multimlopts[i].mlmodel))
            loss_vec_test[i].val = push!(loss_vec_test[i][], loss(netdata[i].test.data..., multimlopts[i].mlmodel))

            # Append predicted model output
            reset!(multimlopts[i].mlmodel)
            yp_train = reduce(vcat,
                [multimlopts[i].mlmodel(xi)[:, end] for
                 xi in netdata[i].train.data[1]]) |> cpu
            yp_test = reduce(vcat,
                [multimlopts[i].mlmodel(xi)[:, end] for
                 xi in netdata[i].test.data[1]]) |> cpu
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