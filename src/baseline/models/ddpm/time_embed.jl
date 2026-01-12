module TimeEmbed

export TimeEmbedding, sinusoidal_embedding, init_time_embedding

using Lux, Random, NNlib

struct TimeEmbedding <: Lux.AbstractLuxLayer
    dim::Int
    hidden_dim::Int
    dense1::Lux.Dense
    dense2::Lux.Dense
end

function init_time_embedding(dim::Int, hidden_dim::Int)
    dense1 = Lux.Dense(dim, hidden_dim, NNlib.gelu)
    dense2 = Lux.Dense(hidden_dim, hidden_dim)
    return TimeEmbedding(dim, hidden_dim, dense1, dense2)
end

function Lux.initialparameters(rng::AbstractRNG, te::TimeEmbedding)
    return (
        dense1 = Lux.initialparameters(rng, te.dense1),
        dense2 = Lux.initialparameters(rng, te.dense2),
    )
end

function Lux.initialstates(rng::AbstractRNG, te::TimeEmbedding)
    return (
        dense1 = Lux.initialstates(rng, te.dense1) |> Lux.f32,
        dense2 = Lux.initialstates(rng, te.dense2) |> Lux.f32,
    )
end

function sinusoidal_embedding(t::AbstractArray{T}, dim::Int) where {T}
    half_dim = dim ÷ 2
    emb_scale = log(10000.0f0) / (half_dim - 1)
    emb = exp.(-(0:(half_dim - 1)) .* emb_scale) |> x -> reshape(x, 1, :)
    emb = t .* emb
    return cat(sin.(emb), cos.(emb), dims = 2)
end

function (te::TimeEmbedding)(t, ps, st)
    emb = sinusoidal_embedding(t, te.dim)
    h, st1 = Lux.apply(te.dense1, emb', ps.dense1, st.dense1)
    h, st2 = Lux.apply(te.dense2, h, ps.dense2, st.dense2)
    return h, (dense1 = st1, dense2 = st2)
end

end
