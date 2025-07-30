module Transformer_Model

export SEQ_Generator, init_SEQ_Generator

using CUDA, Lux, LuxCUDA, ComponentArrays, Accessors, Random, ConfParser
using NNlib: softmax, gelu
using ChainRules.ChainRulesCore: @ignore_derivatives

using ..Utils

if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    include("attention_gpu.jl")
    using .Attention
else
    include("attention.jl")
    using .Attention
end

struct SEQ_Generator <: Lux.AbstractLuxLayer
    depth::Int
    Φ_fcns::NTuple{3,Lux.Dense}
    layernorms::NTuple{2,Lux.LayerNorm}
    attention::NTuple{3,Lux.Dense}
    seq_length::Int
    d_model::Int
    layernorm_bool::Bool
    batchnorm_bool::Bool
end

function init_SEQ_Generator(
    conf::ConfParse,
    x_shape::Tuple,
    rng::AbstractRNG = Random.default_rng(),
)
    prior_widths = (
        try
            parse.(Int, retrieve(conf, "EbmModel", "layer_widths"))
        catch
            parse.(Int, split(retrieve(conf, "EbmModel", "layer_widths"), ","))
        end
    )

    q_size = length(prior_widths) > 2 ? first(prior_widths) : last(prior_widths)

    widths = (
        try
            parse.(Int, retrieve(conf, "GeneratorModel", "widths"))
        catch
            parse.(Int, split(retrieve(conf, "GeneratorModel", "widths"), ","))
        end
    )

    widths = (widths..., first(x_shape))

    first(widths) !== q_size && (error(
        "First expert Φ_hidden_widths must be equal to the hidden dimension of the prior.",
        widths,
        " != ",
        q_size,
    ))

    Φ_functions = Vector{Lux.AbstractLuxLayer}(undef, 0)
    layernorms = Vector{Lux.LayerNorm}(undef, 0)
    attention = Vector{Lux.AbstractLuxLayer}(undef, 0)

    sequence_length = parse(Int, retrieve(conf, "SEQ", "sequence_length"))

    act = gelu
    d_model = parse(Int, retrieve(conf, "SEQ", "d_model"))
    # Projection
    push!(Φ_functions, Lux.Dense(q_size => d_model))
    push!(layernorms, Lux.LayerNorm((d_model, 1), gelu))

    # Query, Key, Value - self-attention
    attention = [
        Lux.Dense(d_model => d_model),
        Lux.Dense(d_model => d_model),
        Lux.Dense(d_model => d_model),
    ]

    # Feed forward
    push!(Φ_functions, Lux.Dense(d_model => d_model))
    push!(layernorms, Lux.LayerNorm((d_model, 1), gelu))

    # Output layer
    push!(Φ_functions, Lux.Dense(d_model => first(x_shape)))
    depth = 3

    return SEQ_Generator(
        depth,
        (Φ_functions...,),
        (layernorms...,),
        (attention...,),
        sequence_length,
        d_model,
        true,
        false,
    )
end

function (gen::SEQ_Generator)(
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    z::AbstractArray{T},
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
    """
    Generate data from the Transformer decoder.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        z: The latent variable.

    Returns:
        The generated data. 
    """
    z = sum(z, dims = 2)

    # Projection
    z, st_new = Lux.apply(gen.Φ_fcns[1], z, ps.fcn[:a], st_lux.fcn[:a])
    @ignore_derivatives @reset st_lux.fcn[:a] = st_new
    z, st_new = Lux.apply(gen.layernorms[1], z, ps.layernorm[:a], st_lux.layernorm[:a])
    @ignore_derivatives @reset st_lux.layernorm[:a] = st_new

    z_prev = z
    for t = 2:gen.seq_length

        # Self-attention
        Q, st_new = Lux.apply(gen.attention[1], z, ps.attention[:Q], st_lux.attention[:Q])
        @ignore_derivatives @reset st_lux.attention[:Q] = st_new
        K, st_new = Lux.apply(gen.attention[2], z, ps.attention[:K], st_lux.attention[:K])
        @ignore_derivatives @reset st_lux.attention[:K] = st_new
        V, st_new = Lux.apply(gen.attention[3], z, ps.attention[:V], st_lux.attention[:V])
        @ignore_derivatives @reset st_lux.attention[:V] = st_new

        attn = scaled_dot_product_attention(Q, K, V, gen.d_model)
        z = z + attn

        # Feed forward
        z, st_new = Lux.apply(gen.Φ_fcns[2], z, ps.fcn[:b], st_lux.fcn[:b])
        @ignore_derivatives @reset st_lux.fcn[:b] = st_new
        z, st_new = Lux.apply(
            gen.layernorms[2],
            z[:, end:end, :],
            ps.layernorm[:b],
            st_lux.layernorm[:b],
        )
        @ignore_derivatives @reset st_lux.layernorm[:b] = st_new

        z = cat(z_prev, z, dims = 2)
        z_prev = z
    end

    # Output layer
    z, st_new = Lux.apply(gen.Φ_fcns[3], z, ps.fcn[:c], st_lux.fcn[:c])
    @ignore_derivatives @reset st_lux.fcn[:c] = st_new

    return z, st_lux
end

end
