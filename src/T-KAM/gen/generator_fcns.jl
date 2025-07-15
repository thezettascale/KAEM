module GeneratorFCNs

export KAN_fwd, CNN_fwd, SEQ_fwd

using CUDA, KernelAbstractions, Accessors, Tullio
using Lux, LuxCUDA
using NNlib: softmax, batched_mul
using ChainRules: @ignore_derivatives

include("../kan/univariate_functions.jl")
include("../../utils.jl")
using .Utils: half_quant, full_quant, device, set_state!
using .UnivariateFunctions: fwd

function KAN_fwd(lkhood, ps, st, z::AbstractArray{T}) where {T<:half_quant}
    """
    Generate data from the KAN likelihood model.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        seed: The seed for the random number generator.

    Returns:
        The generated data.
        The updated seed.
    """
    num_samples = size(z)[end]
    z = dropdims(sum(z, dims = 2), dims = 2)
    new_st = Dict()

    # KAN functions
    for i = 1:lkhood.depth
        z = fwd(lkhood.Φ_fcns[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = dropdims(sum(z, dims = 1); dims = 1)

        if lkhood.layernorm && i < lkhood.depth
            z, st_new = Lux.apply(
                lkhood.Φ_fcns[Symbol("ln_$i")],
                z,
                ps[Symbol("ln_$i")],
                st[Symbol("ln_$i")],
            )
            @ignore_derivatives new_st[Symbol("ln_$i")] = st_new
        end
    end

    @ignore_derivatives set_state!(st, new_st)
    return reshape(z, lkhood.x_shape..., num_samples), st
end

function CNN_fwd(lkhood, ps, st, z::AbstractArray{T}) where {T<:half_quant}
    """
    Generate data from the CNN likelihood model.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        seed: The seed for the random number generator.
    Returns:
        The generated data.
        The updated seed.
    """
    z = reshape(sum(z, dims = 2), 1, 1, first(size(z)), last(size(z)))
    new_st = Dict()

    for i = 1:lkhood.depth
        z, st_new =
            Lux.apply(lkhood.Φ_fcns[Symbol("$i")], z, ps[Symbol("$i")], st[Symbol("$i")])
        @ignore_derivatives new_st[Symbol("$i")] = st_new

        if lkhood.batchnorm
            z, st_new = Lux.apply(
                lkhood.Φ_fcns[Symbol("bn_$i")],
                z,
                ps[Symbol("bn_$i")],
                st[Symbol("bn_$i")],
            )
            @ignore_derivatives new_st[Symbol("bn_$i")] = st_new
        end
    end

    z, st_new = Lux.apply(
        lkhood.Φ_fcns[Symbol("$(lkhood.depth+1)")],
        z,
        ps[Symbol("$(lkhood.depth+1)")],
        st[Symbol("$(lkhood.depth+1)")],
    )
    @ignore_derivatives new_st[Symbol("$(lkhood.depth+1)")] = st_new

    @ignore_derivatives set_state!(st, new_st)
    return z, st
end

function scaled_dot_product_attention(
    Q::AbstractArray{T},
    K::AbstractArray{T},
    V::AbstractArray{T},
    d_model::Int,
) where {T<:half_quant}
    scale = sqrt(eltype(Q)(d_model))
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

function SEQ_fwd(lkhood, ps, st, z::AbstractArray{T}) where {T<:half_quant}
    """
    Generate data from the Transformer decoder.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        z: The latent variable.

    Returns:
        The generated data.
        The updated seed.
    """
    z = sum(z, dims = 2)
    new_st = Dict()

    # Projection
    z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("1")], z, ps[Symbol("1")], st[Symbol("1")])
    @ignore_derivatives new_st[Symbol("1")] = st_new
    z, st_new =
        Lux.apply(lkhood.Φ_fcns[Symbol("ln_1")], z, ps[Symbol("ln_1")], st[Symbol("ln_1")])
    @ignore_derivatives new_st[Symbol("ln_1")] = st_new

    z_prev = z
    for t = 2:lkhood.seq_length

        # Self-attention
        Q, st_new =
            Lux.apply(lkhood.Φ_fcns[Symbol("Q")], z, ps[Symbol("Q")], st[Symbol("Q")])
        @ignore_derivatives new_st[Symbol("Q")] = st_new
        K, st_new =
            Lux.apply(lkhood.Φ_fcns[Symbol("K")], z, ps[Symbol("K")], st[Symbol("K")])
        @ignore_derivatives new_st[Symbol("K")] = st_new
        V, st_new =
            Lux.apply(lkhood.Φ_fcns[Symbol("V")], z, ps[Symbol("V")], st[Symbol("V")])
        @ignore_derivatives new_st[Symbol("V")] = st_new

        attn = scaled_dot_product_attention(Q, K, V, lkhood.d_model)
        z = z .+ attn

        # Feed forward
        z, st_new =
            Lux.apply(lkhood.Φ_fcns[Symbol("2")], z, ps[Symbol("2")], st[Symbol("2")])
        @ignore_derivatives new_st[Symbol("2")] = st_new
        z, st_new = Lux.apply(
            lkhood.Φ_fcns[Symbol("ln_2")],
            z[:, end:end, :],
            ps[Symbol("ln_2")],
            st[Symbol("ln_2")],
        )
        @ignore_derivatives new_st[Symbol("ln_2")] = st_new

        z = cat(z_prev, z, dims = 2)
        z_prev = z
    end

    # Output layer
    z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("3")], z, ps[Symbol("3")], st[Symbol("3")])
    @ignore_derivatives new_st[Symbol("3")] = st_new

    @ignore_derivatives set_state!(st, new_st)
    return z, st
end

end
