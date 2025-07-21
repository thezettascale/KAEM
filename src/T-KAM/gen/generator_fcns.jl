module GeneratorFCNs

export KAN_fwd, CNN_fwd, SEQ_fwd

using CUDA, KernelAbstractions, Accessors, Tullio, ComponentArrays
using Lux, LuxCUDA, ParallelStencil
using NNlib: softmax, batched_mul

include("../kan/univariate_functions.jl")
include("../../utils.jl")
using .Utils: half_quant, full_quant, symbol_map
using .UnivariateFunctions

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

function KAN_fwd(
    lkhood,
    ps::ComponentArray{T},
    st::NamedTuple,
    z::AbstractArray{T},
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
    """
    Generate data from the KAN likelihood model.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.

    Returns:
        The generated data.
    """
    num_samples = size(z)[end]
    z = dropdims(sum(z, dims = 2), dims = 2)

    # KAN functions
    for i = 1:lkhood.depth
        z, st_new =
            Lux.apply(lkhood.Φ_fcns[i], z, ps.fcn[symbol_map[i]], st.fcn[symbol_map[i]])
        z = dropdims(sum(z, dims = 1); dims = 1)

        z, st_new =
            (lkhood.layernorm_bool && i < lkhood.depth) ?
            Lux.apply(
                lkhood.layernorms[i],
                z,
                ps.layernorm[symbol_map[i]],
                st.layernorm[symbol_map[i]],
            ) : (z, st)
        (lkhood.layernorm_bool && i < lkhood.depth) &&
            @reset st.layernorm[symbol_map[i]] = st_new
    end

    return reshape(z, lkhood.x_shape..., num_samples), st
end

function CNN_fwd(
    lkhood,
    ps::ComponentArray{T},
    st::NamedTuple,
    z::AbstractArray{T},
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
    """
    Generate data from the CNN likelihood model.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        rng: The random number generator.
    Returns:
        The generated data.
    """
    z = reshape(sum(z, dims = 2), 1, 1, first(size(z)), last(size(z)))

    for i = 1:(lkhood.depth-1)
        z, st_new =
            Lux.apply(lkhood.Φ_fcns[i], z, ps.fcn[symbol_map[i]], st.fcn[symbol_map[i]])
        @reset st.fcn[symbol_map[i]] = st_new

        z, st_new =
            (lkhood.batchnorm_bool && i < lkhood.depth) ?
            Lux.apply(
                lkhood.batchnorms[i],
                z,
                ps.batchnorm[symbol_map[i]],
                st.batchnorm[symbol_map[i]],
            ) : (z, st)
        (lkhood.batchnorm_bool && i < lkhood.depth) &&
            @reset st.batchnorm[symbol_map[i]] = st_new
    end

    z, st_new = Lux.apply(
        lkhood.Φ_fcns[lkhood.depth],
        z,
        ps.fcn[symbol_map[lkhood.depth]],
        st.fcn[symbol_map[lkhood.depth]],
    )
    @reset st.fcn[symbol_map[lkhood.depth]] = st_new

    return z, st
end

@parallel_indices (d, t, b) function scaled_dot_prod!(
    QK::AbstractArray{T},
    Q::AbstractArray{T},
    K::AbstractArray{T},
    scale::T,
    d_model::Int,
) where {T<:half_quant}
    acc = zero(T)
    for i = 1:d_model
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
    @parallel (1:D, 1:L, 1:B) scaled_dot_prod!(QK, Q, K, scale, d_model)
    QK = softmax(QK, dims = 2)
    @parallel (1:D, 1:L, 1:B) value_kernel!(Q, QK, V, I)
    return Q
end

function SEQ_fwd(
    lkhood,
    ps::ComponentArray{T},
    st::NamedTuple,
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
    z, st_new = Lux.apply(lkhood.Φ_fcns[1], z, ps.fcn[:a], st.fcn[:a])
    @reset st.fcn[:a] = st_new
    z, st_new = Lux.apply(lkhood.layernorms[1], z, ps.layernorm[:a], st.layernorm[:a])
    @reset st.layernorm[:a] = st_new

    z_prev = z
    for t = 2:lkhood.seq_length

        # Self-attention
        Q, st_new = Lux.apply(lkhood.attention[1], z, ps.attention[:Q], st.attention[:Q])
        @reset st.attention[:Q] = st_new
        K, st_new = Lux.apply(lkhood.attention[2], z, ps.attention[:K], st.attention[:K])
        @reset st.attention[:K] = st_new
        V, st_new = Lux.apply(lkhood.attention[3], z, ps.attention[:V], st.attention[:V])
        @reset st.attention[:V] = st_new

        attn = scaled_dot_product_attention(Q, K, V, lkhood.d_model)
        z = z + attn

        # Feed forward
        z, st_new = Lux.apply(lkhood.Φ_fcns[2], z, ps.fcn[:b], st.fcn[:b])
        @reset st.fcn[:b] = st_new
        z, st_new = Lux.apply(
            lkhood.layernorms[2],
            z[:, end:end, :],
            ps.layernorm[:b],
            st.layernorm[:b],
        )
        @reset st.layernorm[:b] = st_new

        z = cat(z_prev, z, dims = 2)
        z_prev = z
    end

    # Output layer
    z, st_new = Lux.apply(lkhood.Φ_fcns[3], z, ps.fcn[:c], st.fcn[:c])
    @reset st.fcn[:c] = st_new

    return z, st
end

end
