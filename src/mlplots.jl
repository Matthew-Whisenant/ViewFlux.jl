"""

    mltrain(multimlopts, dataopts epoch_vec, loss_vec_train, loss_vec_test)

    multimlopts    | Vector of mlopts. See struct mlopts.
    dataopts       | Training and testing data. See struct dataopts.
    netdata        | Data used directly to train network
    epoch_vec      | Vector containing vector of models epoches.
    loss_vec_train | Vector containing vector of models training loss.
    loss_vec_test  | Vector containing vector of models testing loss.

    Create easy to use plot function.

"""
function mlplot(mlopts, dataopts, netdata)

    # Get number of networks
    num_nets = length(mlopts.model)

    # Initial epoch network state
    epoch_vec = Observable([0])
    loss_vec_train = [Observable([loss(netdata.train[i].data...,
        mlopts.model[i])]) for i = 1:num_nets]
    loss_vec_test = [Observable([loss(netdata.test[i].data...,
        mlopts.model[i])]) for i = 1:num_nets]

    # Marker shapes
    truemarker = :diamond
    trainmarker = :circle
    testmarker = :rect

    # Set theme
    set_theme!(theme_black())
    update_theme!(linewidth=3, markersize=15)

    # Create figure
    fig = Figure(resolution=(800, 1000))

    # Axes and labels
    ax = (
        Axis(
            fig[1, 1],
            xlabel="Input X",
            ylabel="Output Y"
        ),
        Axis(
            fig[2, 1],
            xlabel="Epoch",
            ylabel="Loss",
            yscale=log10
        )
    )

    # Make top plot bigger
    rowsize!(fig.layout, 1, Relative(2 / 3))

    # Make easy plotting variables
    xp = reduce(vcat, dataopts.x)
    yp = reduce(vcat, dataopts.y)

    # Epoch slider to control predicted values
    sl = Slider(fig[3, 1], range=0:epoch_vec[][end], startvalue=0)

    # Plot actual data
    scatterlines!(
        fig[1, 1],
        xp, yp;
        marker=truemarker,
        label="Sample Function"
    )

    for i = 1:length(multimlopts)

        # Train and test input
        xp_train = (reduce(vcat, netdata[i].train.data[1])|>cpu)[:, end]
        xp_test = (reduce(vcat, netdata[i].test.data[1])|>cpu)[:, end]

        # Predicted train and test output
        yp_train = reduce(vcat,
            [multimlopts[i].mlmodel(xi)[:, end] for
             xi in netdata[i].train.data[1]]) |> cpu
        yp_test = reduce(vcat,
            [multimlopts[i].mlmodel(xi)[:, end] for
             xi in netdata[i].test.data[1]]) |> cpu

        yp_model = Observable([[yp_train; yp_test]])

        yp_slide = lift(sl.selected_index) do sl_i
            yp_model.val[sl_i]
        end

        markerops = [
            fill(trainmarker, length(yp_train))
            fill(testmarker, length(yp_test))
        ]

        # Predicted output
        scatterlines!(
            fig[1, 1],
            reduce(vcat, [xp_train; xp_test]), yp_slide;
            marker=markerops,
            label="Net " * string(i)
        )

        reset!(multimlopts[i].mlmodel)

        # Show training and testing loss
        lines!(
            fig[2, 1],
            epoch_vec, loss_vec_train[i];
            label="Net " * string(i) * " training"
        )

        lines!(
            fig[2, 1],
            epoch_vec, loss_vec_test[i];
            linestyle=:dash,
            label="Net " * string(i) * " testing")

    end

    Legend(fig[0, 1], fig.content[1], orientation=:horizontal, tellwidth=false, tellheight=true, framevisible=false)

    return fig, yp_model, sl

end