"""

    mldata(mlopts, dataopts)

    Input:
        mlopts   | Machine learning options. See struct mlopts.
        dataopts | Raw data. See struct dataopts.
        mydevice | Cpu or gpu.

    Output:
        netdata | Training and testing data ready for network training.

    Data function. Make data a sequence based on seq_length.
    Puts data on gpu or cpu, then splits data into training and testing.

"""
function mldata(mlopts, dataopts, mydevice)

    # Get number of networks
    num_nets = length(mlopts.model)

    # Allocate data arrays
    train_data = Vector{DataLoader}(undef, num_nets)
    test_data = Vector{DataLoader}(undef, num_nets)

    # Wrap all data into a vector for Flux.jl
    x = map(xᵢ -> [xᵢ], dataopts.x)
    y = map(yᵢ -> [yᵢ], dataopts.y)

    # Data handeled differently for each model based on options
    for i ∈ 1:num_nets

        # Sequence based on seq_length
        s = length(x) - mlopts.seq_length[i] + 1

        # Change input and output data to be
        xtrain = [reduce(hcat, x[t:end-s+t]) for t ∈ 1:s]
        ytrain = y[mlopts.seq_length[i]:end]

        # Make id based on training/testing ratio
        id = round(Int, mlopts.ratio[i] * length(xtrain))

        # Put data in tuple for training
        train_data[i] = DataLoader(
            (
                xtrain[1:id] |> mydevice,
                ytrain[1:id] |> mydevice
            ),
            batchsize=mlopts.batch_size[i],
            shuffle=mlopts.shuf_data[i]
        )

        test_data[i] = DataLoader(
            (
                xtrain[id+1:end] |> mydevice,
                ytrain[id+1:end] |> mydevice
            ),
            batchsize=mlopts.batch_size[i],
            shuffle=mlopts.shuf_data[i]
        )

    end

    netdata = (train=train_data, test=test_data)

    return netdata

end