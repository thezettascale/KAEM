module Transformer_Model

export SEQ_Generator, init_SEQ_Generator

using Lux, ComponentArrays, Accessors, Random, ConfParser
using NNlib: softmax, gelu, batched_mul

using ..Utils

struct BoolConfig <: AbstractBoolConfig
    layernorm::Bool
    batchnorm::Bool
end

struct SEQ_Generator <: Lux.AbstractLuxLayer
    depth::Int
    Φ_fcns::Tuple{Vararg{Lux.Dense}}
    layernorms::Tuple{Vararg{Lux.LayerNorm}}
    attention::Tuple{Vararg{Lux.Dense}}
    seq_length::Int
    d_model::Int
    bool_config::BoolConfig
    s_size::Int
end

function init_SEQ_Generator(
        conf::ConfParse,
        x_shape::Tuple,
        rng::AbstractRNG = Random.default_rng(),
    )
    prior_widths = parse_config_array(Int, retrieve(conf, "EbmModel", "layer_widths"))
    q_size = get_q_size(prior_widths)
    widths = parse_config_array(Int, retrieve(conf, "GeneratorModel", "widths"))
    widths = (widths..., first(x_shape))
    validate_generator_widths(widths, q_size)

    Φ_functions = Vector{Lux.AbstractLuxLayer}(undef, 0)
    layernorms = Vector{Lux.LayerNorm}(undef, 0)
    attention = Vector{Lux.AbstractLuxLayer}(undef, 0)

    sequence_length = parse(Int, retrieve(conf, "SEQ", "sequence_length"))

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

    s_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    return SEQ_Generator(
        depth,
        Tuple(Φ_functions),
        Tuple(layernorms),
        Tuple(attention),
        sequence_length,
        d_model,
        BoolConfig(true, false),
        s_size
    )
end

function scaled_dotprod_attn(
        Q,
        K,
        V,
        d_model,
    )
    scale = sqrt(Float32(d_model))

    QK = batched_mul(permutedims(Q, (2, 1, 3)), K)
    QK ./= scale
    QK = softmax(QK, dims = 2)

    attn = batched_mul(V, permutedims(QK, (2, 1, 3)))
    return attn
end

function (gen::SEQ_Generator)(
        ps,
        st_kan,
        st_lux,
        z,
    )
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
    st_lux_new = st_lux

    # Projection
    z, st_layer_new = Lux.apply(gen.Φ_fcns[1], z, ps.fcn[:a], st_lux_new.fcn[:a])
    @reset st_lux_new.fcn[:a] = st_layer_new
    z, st_layer_new = Lux.apply(gen.layernorms[1], z, ps.layernorm[:a], st_lux_new.layernorm[:a])
    @reset st_lux_new.layernorm[:a] = st_layer_new

    z_prev = z
    for t in 2:gen.seq_length

        # Self-attention
        Q, st_layer_new = Lux.apply(gen.attention[1], z, ps.attention[:Q], st_lux_new.attention[:Q])
        @reset st_lux_new.attention[:Q] = st_layer_new
        K, st_layer_new = Lux.apply(gen.attention[2], z, ps.attention[:K], st_lux_new.attention[:K])
        @reset st_lux_new.attention[:K] = st_layer_new
        V, st_layer_new = Lux.apply(gen.attention[3], z, ps.attention[:V], st_lux_new.attention[:V])
        @reset st_lux_new.attention[:V] = st_layer_new

        attn = scaled_dotprod_attn(Q, K, V, gen.d_model)
        z = @. z + attn

        # Feed forward
        z, st_layer_new = Lux.apply(gen.Φ_fcns[2], z, ps.fcn[:b], st_lux_new.fcn[:b])
        @reset st_lux_new.fcn[:b] = st_layer_new
        z, st_layer_new = Lux.apply(
            gen.layernorms[2],
            z[:, end:end, :],
            ps.layernorm[:b],
            st_lux_new.layernorm[:b],
        )
        @reset st_lux_new.layernorm[:b] = st_layer_new

        z = cat(z_prev, z, dims = 2)
        z_prev = z
    end

    # Output layer
    z, st_layer_new = Lux.apply(gen.Φ_fcns[3], z, ps.fcn[:c], st_lux_new.fcn[:c])
    @reset st_lux_new.fcn[:c] = st_layer_new

    return z, st_lux_new
end

end
