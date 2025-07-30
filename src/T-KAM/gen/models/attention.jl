module Attention

export scaled_dot_product_attention

using ..Utils

using CUDA, ParallelStencil
using NNlib: softmax

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

end