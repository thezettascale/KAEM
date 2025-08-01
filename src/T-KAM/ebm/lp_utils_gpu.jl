
module LogPriorUtils

export log_norm, log_alpha, log_mix_pdf

using ..Utils

using NNlib: softmax
using CUDA, Lux, LuxCUDA, LinearAlgebra, Accessors, Random, ComponentArrays
using KernelAbstractions, Tullio

function log_norm(norm::AbstractArray{T}, ε::T)::AbstractArray{T} where {T<:half_quant}
    @tullio Z[q, p] := norm[q, p, g]
    return log.(Z .+ ε)
end

function log_alpha(
    log_απ::AbstractArray{T},
    alpha::AbstractArray{T},
    ε::T,
    Q::Int,
    P::Int,
    S::Int,
)::AbstractArray{T} where {T<:half_quant}
    return log.(log_απ .+ alpha .+ ε)
end

function log_mix_pdf(
    f::AbstractArray{T},
    log_απ::AbstractArray{T},
    log_Z::AbstractArray{T},
    reg::T,
    Q::Int,
    P::Int,
    S::Int,
)::AbstractArray{T} where {T<:half_quant}
    @tullio lp[s] := f[q, p, s] + log_απ[q, p, s] - log_Z[q, p]
    return lp .+ reg
end

end
