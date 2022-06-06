# * Simple Periodic ML Model

# Get network and plotting function
include(abspath("modules/Viewflux.jl"))
using .Viewflux

# Plotting options
plotopts = Viewflux.plotopts(1, 20, "sinnet.gif")

# Training data options
actual(x) = sin.(x ./ 2) .+ 4
x_train = [[Float32(i)] for i = 0:49] |> Flux.gpu
x_test = [[Float32(i)] for i = 50:69] |> Flux.gpu
y_train = [actual(x) for x in x_train] |> Flux.gpu
y_test = [actual(x) for x in x_test] |> Flux.gpu
dataopts = Viewflux.dataopts(x_train, x_test, y_train, y_test)

# First network
mlmodel1 = Flux.Chain(
    Flux.Dense(1 => 5, sin),
    Flux.Dense(5 => 1)
) |> Flux.gpu
loss1(x, y) = sum(Flux.Losses.mse(mlmodel1(xi), yi) for (xi, yi) in zip(x, y))
opt1 = Flux.Optimiser(Flux.ADAGrad(1.0f-1))

mlopts1 = Viewflux.mlopts(200, 1, false, mlmodel1, opt1, loss1)

# Second network
mlmodel2 = Flux.Chain(
    Flux.Dense(1 => 5, (x) -> sin(x) + cos(x)),
    Flux.Dense(5 => 1)
) |> Flux.gpu
loss2(x, y) = sum(Flux.Losses.mse(mlmodel2(xi), yi) for (xi, yi) in zip(x, y))
opt2 = Flux.Optimiser(Flux.ADAGrad(1.0f-1))

mlopts2 = Viewflux.mlopts(200, 1, false, mlmodel2, opt2, loss2)

# Training and visualize
mltrain([mlopts1, mlopts2], dataopts, plotopts)

return nothing