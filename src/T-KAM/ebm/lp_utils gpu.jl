
module LogPriorUtils

export log_norm, log_alpha, log_mix_pdf

using NNlib: softmax
using CUDA, Lux, LuxCUDA, LinearAlgebra, Accessors, Random, ComponentArrays

function log_norm(norm::AbstractArray{T}, ε::T)::AbstractArray{T} where {T<:half_quant}
    return log.(dropdims(sum(norm; dims = 3); dims = 3) .+ ε)
end

function log_alpha(
    log_απ::AbstractArray{T},
    alpha::AbstractArray{T},
    ε::T,
    Q::Int,
    P::Int,
    S::Int,
)::AbstractArray{T} where {T<:half_quant}
    return log.(reshape(alpha, size(alpha)..., 1) .* log_απ .+ ε)
end

function log_mix_pdf(
    f::AbstractArray{T},
    log_απ::AbstractArray{T},
    log_Z::AbstractArray{T},
    reg::T,
    Q::Int,
    P::Int,
    S::Int,
)
    logprob = f + log_απ .- log_Z
    return dropdims(sum(logprob; dims = (1, 2)); dims = (1, 2)) .+ reg
end

end
