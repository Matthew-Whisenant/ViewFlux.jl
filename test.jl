using Flux, GLMakie, Random, CUDA
using Flux.Losses: mse
using Flux: @epochs

has_cuda() && CUDA.allowscalar(false)
device = gpu

Random.seed!(1)
seq_length = 1

x = [[Float32(i)] for i ∈ range(0, 4 * π, 100)]
y = [sin.(i) .+ i .* 0.2 .+ 4 for i ∈ x]

s = length(x) - seq_length + 1

xtrain = [reduce(hcat, x[t:end-s+t]) for t ∈ 1:s]
ytrain = y[seq_length:end]

data = Flux.Data.DataLoader((xtrain, ytrain) |> device)

m = let inner = 20
    Chain(
        Dense(1 => inner, sin),
        LSTM(inner => inner),
        LSTM(inner => inner),
        Dense(inner => 1)
    ) |> device
end

ps = Flux.params(m)
η = 0.1f0
opt = Flux.Optimiser(Descent(η))

function loss(x, y)
    #Flux.reset!(m)
    sum(mse(m(xi)[:, end], yi) for (xi, yi) ∈ zip(x, y))
end

fig, ax, lin1 = scatterlines(reduce(vcat, x), reduce(vcat, y))
yp = Observable(reduce(vcat, [(m(xi))[:, end] for xi ∈ data.data[1]] |> cpu))
lin2 = scatterlines!(ax, reduce(vcat, x)[seq_length:end], yp)
display(fig)

for i ∈ 1:100

    for d in data

        grads = Flux.gradient(ps) do
            loss(d...)
        end

        Flux.Optimise.update!(opt, ps, grads)

    end

    opt.os[1].eta = opt.os[1].eta * 0.99
    Flux.reset!(m)

    if mod(i, 10) == 0
        @info loss(data.data...)
        Flux.reset!(m)
        global yp[] = reduce(vcat, [(m(xi))[:, end] for xi ∈ data.data[1]] |> cpu)
        Flux.reset!(m)
        autolimits!(ax)
        sleep(1 / 30)
    end

end