
module LogPriorUtils

export log_norm, log_alpha, log_mix_pdf

using ..Utils

using NNlib: softmax
using CUDA, Lux, LuxCUDA, LinearAlgebra, Accessors, Random, ComponentArrays
using KernelAbstractions, Tullio

function log_norm(norm::AbstractArray{T,3}, ε::T)::AbstractArray{T,2} where {T<:half_quant}
    return dropdims(log.(sum(norm, dims = 3) .+ ε), dims = 3)
end

function log_mix_pdf(
    f::AbstractArray{T,3},
    α::AbstractArray{T,2},
    π_0::AbstractArray{T,3},
    Z::AbstractArray{T,2},
    ε::T,
    Q::Int,
    P::Int,
    S::Int,
)::AbstractArray{T,1} where {T<:half_quant}
    @tullio lp[q, s] := exp(f[q, p, s]) * π_0[q, 1, s] * α[q, p] / Z[q, p]
    lp = lp .+ ε
    @tullio out[s] := log(lp[q, s])
    return out
end

end
