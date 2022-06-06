"""

    mydevice = mldevice(mlopts, dataopts)

    Input:
        mlopts   | Machine learning options. See struct mlopts.
        dataopts | Raw data. See struct dataopts.

    Output:
        mydevice | CPU or GPU for data loading.

    Training function. Automatically does plotting according to
    variables set in plotopts.

"""
function mldevice(mlopts, dataopts)

    # Get device to use
    (dataopts.use_cuda && CUDA.functional()) ? mydevice = gpu : mydevice = cpu

    # Set models on device
    mlopts.model = mlopts.model .|> mydevice

    return mydevice

end