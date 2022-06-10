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
function mlplot(mlopts, dataopts, netdata, loss)

    # Get number of networks
    num_nets = length(mlopts.model)

    # Create tuple to hold vectors observables
    obs =
        let VF = Vector{Float64}, VI = Vector{Int}
            (
                # Initialize observables
                idx=Observable{Int}(1),

                # Predicted output
                yp_model=[Observable{Vector{VF}}([]) for i = 1:num_nets],

                # Epoch list for slider
                epoch_vec=Observable{VI}([0]),

                # Loss list
                loss_vec_train=[Observable{VF}([]) for i = 1:num_nets],
                loss_vec_test=[Observable{VF}([]) for i = 1:num_nets]
            )
        end

    # Set theme
    with_theme(theme_black(), linewidth=3, markersize=15) do

        # Marker shapes
        truemarker = :diamond
        trainmarker = :circle
        testmarker = :rect

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

        # Plot actual data
        scatterlines!(
            fig[1, 1],
            reduce(vcat, dataopts.x), reduce(vcat, dataopts.y);
            marker=truemarker,
            label="Sample Function"
        )

        for i = 1:num_nets

            # Train and test input
            xp_train = (reduce(vcat, netdata.train[i].data[1])|>cpu)[:, end]
            xp_test = (reduce(vcat, netdata.test[i].data[1])|>cpu)[:, end]

            xp = reduce(vcat, [xp_train; xp_test])

            # Predicted train and test output
            yp_train = reduce(vcat,
                [mlopts.model[i](xi)[:, end] for
                 xi in netdata.train[i].data[1]]) |> cpu
            yp_test = reduce(vcat,
                [mlopts.model[i](xi)[:, end] for
                 xi in netdata.test[i].data[1]]) |> cpu

            push!(obs.yp_model[i][], [yp_train; yp_test])

            # Append network losses
            reset!(mlopts.model[i])
            push!(obs.loss_vec_train[i][], loss(netdata.train[i].data..., mlopts.model[i]))
            push!(obs.loss_vec_test[i][], loss(netdata.test[i].data..., mlopts.model[i]))
            reset!(mlopts.model[i])

            # Predicted output
            markerops = [
                fill(trainmarker, length(yp_train))
                fill(testmarker, length(yp_test))
            ]

            # Predicted output
            scatterlines!(
                fig[1, 1],
                reduce(vcat, xp),
                @lift(reduce(vcat, $(obs.yp_model[i])[$(obs.idx)]));
                marker=markerops,
                label="Net " * string(i)
            )

            # Show training and testing loss
            lines!(
                fig[2, 1],
                obs.epoch_vec, obs.loss_vec_train[i];
                color=Cycled(i + 1),
                label="Net " * string(i) * " training"
            )

            lines!(
                fig[2, 1],
                obs.epoch_vec, obs.loss_vec_test[i];
                linestyle=:dash,
                color=Cycled(i + 1),
                label="Net " * string(i) * " testing")

        end

        # Epoch slider to control predicted values
        sl = Slider(fig[3, 1], range=@lift(0:$(obs.epoch_vec)[end]), startvalue=0)
        connect!(obs.idx, sl.selected_index)

        Legend(fig[0, 1], fig.content[1], orientation=:horizontal, tellwidth=false, tellheight=true, framevisible=false)

        return fig, obs, sl

    end

end