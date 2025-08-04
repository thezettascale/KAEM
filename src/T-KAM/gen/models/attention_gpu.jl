module Attention

export scaled_dot_product_attention

using ..Utils

using CUDA, KernelAbstractions, Tullio
using NNlib: softmax

function scaled_dot_product_attention(
    Q::AbstractArray{T,3},
    K::AbstractArray{T,3},
    V::AbstractArray{T,3},
    d_model::Int,
)::AbstractArray{T,3} where {T<:half_quant}
    D, L, B = size(Q)
    I = size(K, 2)
    scale = sqrt(T(d_model))

    @tullio QK[t, i, b] := Q[d, t, b] * K[d, i, b]
    QK = softmax(QK, dims = 2)
    @tullio attn[d, t, b] := QK[t, i, b] * V[d, i, b]
    return attn
end

end
