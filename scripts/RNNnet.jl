# * Simple Linear ML Model

# Import ML and plotting libraries
import Flux, Plots, CUDA, ProgressMeter

#? -------------- DEBUG SETTINGS --------------

# Disallow slow scalar gpu indexing
CUDA.has_cuda() && CUDA.allowscalar(false)

# Debug messages (on:"mltrain", off:"")
ENV["JULIA_DEBUG"] = ""

#? -------------- PLOT SETTINGS --------------

# # Set plot theme
Plots.theme(:dracula)

# Epoch inverval to plot on (1 is every epoch, 5 is every 5 epochs)
epoch_plot = 10

# Frames per second for gif (less is slower graph)
fps = 10

# Name of gif
filename = "RNNnet.gif"

#? -------------- TRAINING DATA GENERATION --------------

# Simple function
actual(x) = sin.(x ./ 2) .+ 0.1 .* x

# Create network from input layers
nlayers = 20
predict = Flux.Chain(
    Flux.Dense(1 => nlayers, x -> sin(x) + x),
    Flux.LSTM(nlayers => nlayers),
    Flux.Dense(nlayers => 1)
) |> Flux.gpu

# Training input data
x_train = [[Float32(i)] for i = 0:49]

# Testing input data
x_test = [[Float32(i)] for i = 50:69]

# Training output data
y_train = [actual(x) for x in x_train]

# Testing output data
y_test = [actual(x) for x in x_test]

#? -------------- TRAINING SETTINGS --------------

# Number of training epochs
num_epochs = 1000

# Network Optimiser
opt = Flux.Optimise.Optimiser(
    Flux.Optimise.AMSGrad(1.0f-3)
)
# Batch size
batch_size = 1

# Shuffle data
shuf_data = false

# Loss function with mean square error and L2 regularzation
loss(x, y) = sum(Flux.Losses.mse(predict(xi), yi) for (xi, yi) in zip(x, y))

#? -------------- RUN FUNCTIONS --------------

# Get network and plotting function
include.(readdir(abspath("src"), join=true))

# Do training
mltrain(
    predict, x_train, x_test, y_train, y_test,
    num_epochs, opt, batch_size, shuf_data, loss,
    epoch_plot, fps, filename
)

return nothing