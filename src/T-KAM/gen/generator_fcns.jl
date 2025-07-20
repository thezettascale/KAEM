module GeneratorFCNs

export KAN_fwd, CNN_fwd, SEQ_fwd

using CUDA, KernelAbstractions, Accessors, Tullio, ComponentArrays
using Lux, LuxCUDA
using NNlib: softmax, batched_mul

include("../kan/univariate_functions.jl")
include("../../utils.jl")
using .Utils: half_quant, full_quant, device, set_state!, symbol_map
using .UnivariateFunctions

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
    layernorm_st = Dict()

    # KAN functions
    for i = 1:lkhood.depth
        z, st_new = lkhood.Φ_fcns[i](z, ps.fcn[symbol_map[i]], st.fcn[symbol_map[i]])
        z = dropdims(sum(z, dims = 1); dims = 1)

        if lkhood.layernorm_bool && i < lkhood.depth
            z, st_new = Lux.apply(
                lkhood.layernorms[i],
                z,
                ps.layernorm[symbol_map[i]],
                st.layernorm[symbol_map[i]],
            )
            layernorm_st[symbol_map[i]] = st_new
        end
    end

    set_state!(st.layernorm, layernorm_st)
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
    fcn_st = Dict()
    batchnorm_st = Dict()

    for i = 1:lkhood.depth-1
        z, st_new =
            Lux.apply(lkhood.Φ_fcns[i], z, ps.fcn[symbol_map[i]], st.fcn[symbol_map[i]])
        fcn_st[symbol_map[i]] = st_new

        if lkhood.batchnorm_bool
            z, st_new = Lux.apply(
                lkhood.batchnorms[i],
                z,
                ps.batchnorm[symbol_map[i]],
                st.batchnorm[symbol_map[i]],
            )
            batchnorm_st[symbol_map[i]] = st_new
        end
    end

    z, st_new = Lux.apply(
        lkhood.Φ_fcns[lkhood.depth],
        z,
        ps.fcn[symbol_map[lkhood.depth]],
        st.fcn[symbol_map[lkhood.depth]],
    )
    fcn_st[symbol_map[lkhood.depth]] = st_new

    set_state!(st.fcn, fcn_st)
    set_state!(st.batchnorm, batchnorm_st)
    return z, st
end

function scaled_dot_product_attention(
    Q::AbstractArray{T},
    K::AbstractArray{T},
    V::AbstractArray{T},
    d_model::Int,
) where {T<:half_quant}
    scale = sqrt(T(d_model))
    D, L, B = size(Q)
    _, I, _ = size(K)

    # 1. Compute QK: (T, I, B)
    Qt = permutedims(Q, (2, 1, 3))
    Kt = permutedims(K, (2, 1, 3))
    Qt_ = reshape(Qt, L, 1, D, B)
    Kt_ = reshape(Kt, 1, I, D, B)
    QK = sum(Qt_ .* Kt_, dims = 3)
    QK = dropdims(QK, dims = 3) ./ scale

    # 2. Softmax over I (keys) dimension
    QK_max = maximum(QK, dims = 2)
    QK_exp = exp.(QK .- QK_max)
    QK_softmax = QK_exp ./ sum(QK_exp, dims = 2)

    # 3. Weighted sum: out[d, t, b] = sum_i QK_softmax[t, i, b] * V[d, i, b]
    QK_broad = reshape(QK_softmax, 1, L, I, B)
    V_broad = reshape(V, D, 1, I, B)
    out = sum(QK_broad .* V_broad, dims = 3)
    out = dropdims(out, dims = 3)
    return out
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
    fcn_st = Dict()
    layernorm_st = Dict()
    attention_st = Dict()

    # Projection
    z, st_new = Lux.apply(lkhood.Φ_fcns[1], z, ps.fcn[:a], st.fcn[:a])
    fcn_st[:a] = st_new
    z, st_new =
        Lux.apply(lkhood.layernorms[1], z, ps.layernorm[:a], st.layernorm[:a])
    layernorm_st[:a] = st_new

    z_prev = z
    for t = 2:lkhood.seq_length

        # Self-attention
        Q, st_new =
            Lux.apply(lkhood.attention.Q, z, ps.attention[:Q], st.attention[:Q])
        attention_st[:Q] = st_new
        K, st_new =
            Lux.apply(lkhood.attention.K, z, ps.attention[:K], st.attention[:K]) 
        attention_st[:K] = st_new
        V, st_new =
            Lux.apply(lkhood.attention.V, z, ps.attention[:V], st.attention[:V])
        attention_st[:V] = st_new

        attn = scaled_dot_product_attention(Q, K, V, lkhood.d_model)
        z = z .+ attn

        # Feed forward
        z, st_new =
            Lux.apply(lkhood.Φ_fcns[2], z, ps.fcn[:b], st.fcn[:b])
        fcn_st[:b] = st_new
        z, st_new = Lux.apply(
            lkhood.layernorms[2],
            z[:, end:end, :],
            ps.layernorm[:b],
            st.layernorm[:b],
        )
        layernorm_st[:b] = st_new

        z = cat(z_prev, z, dims = 2)
        z_prev = z
    end

    # Output layer
    z, st_new = Lux.apply(lkhood.Φ_fcns[3], z, ps.fcn[:c], st.fcn[:c])
    fcn_st[:c] = st_new

    set_state!(st.fcn, fcn_st)
    set_state!(st.layernorm, layernorm_st)  
    set_state!(st.attention, attention_st)
    return z, st
end

end
