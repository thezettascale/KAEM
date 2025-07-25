module Transformer_Model

export SEQ_Generator, init_SEQ_Generator

using CUDA, Lux, LuxCUDA, ComponentArrays, Accessors, Random, ConfParser, ParallelStencil
using NNlib: softmax, gelu

using ..Utils

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

@parallel_indices (t, i, b) function scaled_dot_prod!(
    QK::AbstractArray{T},
    Q::AbstractArray{T},
    K::AbstractArray{T},
    scale::T,
    d_model::Int,
) where {T<:half_quant}
    acc = zero(T)
    for d = 1:d_model
        acc = acc + Q[d, t, b] * K[d, i, b]
    end
    QK[t, i, b] = acc / scale
    return nothing
end

@parallel_indices (d, t, b) function value_kernel!(
    out::AbstractArray{T},
    QK::AbstractArray{T},
    V::AbstractArray{T},
    I::Int,
) where {T<:half_quant}
    acc = zero(T)
    for i = 1:I
        acc = acc + QK[t, i, b] * V[d, i, b]
    end
    out[d, t, b] = acc
    return nothing
end

function scaled_dot_product_attention(
    Q::AbstractArray{T},
    K::AbstractArray{T},
    V::AbstractArray{T},
    d_model::Int,
) where {T<:half_quant}
    D, L, B = size(Q)
    I = size(K, 2)
    scale = sqrt(T(d_model))

    QK = @zeros(L, I, B)
    @parallel (1:L, 1:I, 1:B) scaled_dot_prod!(QK, Q, K, scale, d_model)
    QK = softmax(QK, dims = 2)
    @parallel (1:D, 1:L, 1:B) value_kernel!(Q, QK, V, I)
    return Q
end

struct SEQ_Generator <: Lux.AbstractLuxLayer
    depth::Int
    Φ_fcns::Tuple{Lux.Dense}
    layernorms::Tuple{Lux.LayerNorm}
    attention::Tuple{Lux.Dense}
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
    @reset st_lux.fcn[:a] = st_new
    z, st_new = Lux.apply(gen.layernorms[1], z, ps.layernorm[:a], st_lux.layernorm[:a])
    @reset st_lux.layernorm[:a] = st_new

    z_prev = z
    for t = 2:gen.seq_length

        # Self-attention
        Q, st_new = Lux.apply(gen.attention[1], z, ps.attention[:Q], st_lux.attention[:Q])
        @reset st_lux.attention[:Q] = st_new
        K, st_new = Lux.apply(gen.attention[2], z, ps.attention[:K], st_lux.attention[:K])
        @reset st_lux.attention[:K] = st_new
        V, st_new = Lux.apply(gen.attention[3], z, ps.attention[:V], st_lux.attention[:V])
        @reset st_lux.attention[:V] = st_new

        attn = scaled_dot_product_attention(Q, K, V, gen.d_model)
        z = z + attn

        # Feed forward
        z, st_new = Lux.apply(gen.Φ_fcns[2], z, ps.fcn[:b], st_lux.fcn[:b])
        @reset st_lux.fcn[:b] = st_new
        z, st_new = Lux.apply(
            gen.layernorms[2],
            z[:, end:end, :],
            ps.layernorm[:b],
            st_lux.layernorm[:b],
        )
        @reset st_lux.layernorm[:b] = st_new

        z = cat(z_prev, z, dims = 2)
        z_prev = z
    end

    # Output layer
    z, st_new = Lux.apply(gen.Φ_fcns[3], z, ps.fcn[:c], st_lux.fcn[:c])
    @reset st_lux.fcn[:c] = st_new

    return z, st_lux
end

end
