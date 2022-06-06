"""

    mlopts(model, opt, batch_size, shuf_data, ratio, seq_length)

        model      | Vector{Chain}     | Create network from Chain input
        opt        | Vector{Optimiser} | Network optimiser
        batch_size | Vector{Int64}     | Batch size for training
        shuf_data  | Vector{Bool}      | Shuffle training data
        ratio      | Vector{Float64}   | Training to testing data ratio
        seq_length | Vector{Int64}     | Length of sequence input data

    Holds all network related options. Multiple networks can be held as vectors.

"""
mutable struct mlopts_struct

    model::Vector{Chain}
    opt::Vector{Optimiser}
    batch_size::Vector{Int64}
    shuf_data::Vector{Bool}
    ratio::Vector{Float64}
    seq_length::Vector{Int64}

end

"""

    dataopts(x, y, use_cuda)

        x        | Vector{Float32} | Input data
        y        | Vector{Float32} | Output data
        use_cuda | Bool            | Bool to use gpu or cpu

    Holds raw data used for training and testing. Will be put
    on GPU if use_cuda is true and GPU is available.

"""
struct dataopts_struct

    x::Vector{Float32}
    y::Vector{Float32}
    use_cuda::Bool

end

"""

    plotopts(num_epochs, epoch_plot, fps, gifname)

        num_epochs | Int64  | Number of training epochs
        epoch_plot | Int64  | Epoch inverval to plot on (1 is every epoch, 5 is every 5 epochs)
        fps        | Int64  | Frames per second for gif (less is slower graph)
        gifname    | String | Name of gif

    Holds plotting related options.

"""
struct plotopts_struct

    num_epochs::Int64
    epoch_plot::Int64
    fps::Int64
    gifname::String

end