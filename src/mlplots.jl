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
                idvec=Observable{Int}(1),

                # Predicted output
                yp_model=[Observable{Vector{Vector{VF}}}([]) for i = 1:num_nets],

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
            @lift(getindex.(dataopts.y, $(obs.idvec)));
            marker=truemarker,
            label="Sample Function"
        )

        for i = 1:num_nets

            # Predicted train and test output
            yp_train = reduce.(vcat,
                [mlopts.model[i](xi)[:, end]
                 for xi in netdata.train[i].data[1]] |> cpu)
            yp_test = reduce.(vcat,
                [mlopts.model[i](xi)[:, end]
                 for xi in netdata.test[i].data[1]] |> cpu)

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
                [1:length(@lift(getindex.($(obs.yp_model[i])[$(obs.idx)], $(obs.idvec))).val);] .+ mlopts.seq_length[i],
                @lift(getindex.($(obs.yp_model[i])[$(obs.idx)], $(obs.idvec)));
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

        # Add vertical line for epoch plot
        vlines!(ax[2], @lift($(obs.idx) - 1), linewidth=1, color=(:white, 0.3))

        # Epoch slider to control predicted values
        sl1 = SliderGrid(fig[3, 1],
            (
                label="Epoch",
                range=@lift(0:$(obs.epoch_vec)[end]),
                startvalue=0,
                format="{:4d}"
            )
        )
        connect!(obs.idx, sl1.sliders[1].selected_index)

        # Observable slider to control predicted values
        sl2 = SliderGrid(fig[4, 1],
            (
                label="Observable",
                range=1:size(dataopts.y[1], 1),
                startvalue=1,
                format="{:4d}"
            )
        )
        connect!(obs.idvec, sl2.sliders[1].value)

        Legend(fig[0, 1], fig.content[1], orientation=:horizontal, tellwidth=false, tellheight=true, framevisible=false)

        sl = [sl1, sl2]

        return fig, obs, sl

    end

end