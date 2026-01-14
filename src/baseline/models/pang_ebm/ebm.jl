module LatentEBM

using Lux, Random, Accessors, NNlib

using ..Utils

export EnergyMLP, init_energy_mlp

struct EnergyMLP <: Lux.AbstractLuxLayer
    depth::Int
    layers::Tuple{Vararg{Lux.Dense}}
    latent_dim::Int
end

function init_energy_mlp(latent_dim::Int, energy_widths::Vector{Int})
    energy_layers = Vector{Lux.Dense}()
    prev_dim = latent_dim

    for (i, w) in enumerate(energy_widths)
        is_last = i == length(energy_widths)
        if is_last
            push!(energy_layers, Lux.Dense(prev_dim, 1))
        else
            push!(energy_layers, Lux.Dense(prev_dim, w, NNlib.swish))
        end
        prev_dim = w
    end

    return EnergyMLP(length(energy_layers), Tuple(energy_layers), latent_dim)
end

function Lux.initialparameters(rng::AbstractRNG, ebm::EnergyMLP)
    return NamedTuple(
        symbol_map[i] => Lux.initialparameters(rng, ebm.layers[i])
            for i in 1:ebm.depth
    )
end

function Lux.initialstates(rng::AbstractRNG, ebm::EnergyMLP)
    return NamedTuple(
        symbol_map[i] => Lux.initialstates(rng, ebm.layers[i]) |> Lux.f32
            for i in 1:ebm.depth
    )
end

function (ebm::EnergyMLP)(z, ps, st)
    h = z
    st_new = st

    for i in 1:ebm.depth
        h, st_layer = Lux.apply(ebm.layers[i], h, ps[symbol_map[i]], st[symbol_map[i]])
        @reset st_new[symbol_map[i]] = st_layer
    end

    return dropdims(h; dims = 1), st_new
end

end
