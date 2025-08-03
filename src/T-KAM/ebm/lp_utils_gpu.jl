
module LogPriorUtils

export log_norm, log_alpha, log_mix_pdf

using ..Utils

using NNlib: softmax
using CUDA, Lux, LuxCUDA, LinearAlgebra, Accessors, Random, ComponentArrays
using KernelAbstractions, Tullio

function log_norm(norm::AbstractArray{T,3}, ε::T)::AbstractArray{T,2} where {T<:half_quant}
    return dropdims(log.(sum(norm, dims = 3) .+ ε), dims = 3)
end

function log_alpha(
    log_απ::AbstractArray{T,3},
    alpha::AbstractArray{T,2},
    ε::T,
    Q::Int,
    P::Int,
    S::Int,
)::AbstractArray{T,3} where {T<:half_quant}
    return log.(log_απ .+ alpha .+ ε)
end

function log_mix_pdf(
    f::AbstractArray{T,3},
    log_απ::AbstractArray{T,3},
    log_Z::AbstractArray{T,2},
    reg::T,
    Q::Int,
    P::Int,
    S::Int,
)::AbstractArray{T,1} where {T<:half_quant}
    @tullio lp[s] := f[q, p, s] + log_απ[q, p, s] - log_Z[q, p]
    return lp .+ reg
end

end
