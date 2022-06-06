"""

    mldebug(dataopts, multimlopts...)

    Debug training info. Multiple mlopts can be input via a vector.

"""
function mldebug(dataopts, multimlopts...)

    # Show inputs and outputs
    @debug "Input" dataopts.x_train dataopts.x_test
    @debug "Output" dataopts.y_train dataopts.y_test

    # Show train data shape
    # @debug "Number of observations" size(collect(train_data), 1)
    # @debug "Number of features" [size(collect(collect(train_data)[i][1]), 1) for i = 1:length(train_data)]
    # @debug "Number of observations" [size(collect(collect(train_data)[i][1]), 2) for i = 1:length(train_data)]

    # Show network structure
    for mlopts in multimlopts

        @debug "Network Weights" [mlopts.predict.layers[i].weight for i = 1:length(mlopts.predict)]
        @debug "Network Biases" [mlopts.predict.layers[i].bias for i = 1:length(mlopts.predict)]
        @debug "Optimisers" mlopts.opt

    end

    return nothing

end