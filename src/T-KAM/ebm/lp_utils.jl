
module LogPriorUtils

export log_norm, log_alpha, log_mix_pdf

using ..Utils

using NNlib: softmax
using CUDA, Lux, LuxCUDA, LinearAlgebra, Accessors, Random, ComponentArrays, ParallelStencil

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, half_quant, 3)
else
    @init_parallel_stencil(Threads, half_quant, 3)
end

@parallel_indices (q, p) function log_norm_kernel!(
    log_Z::AbstractArray{T,2},
    norm::AbstractArray{T,3},
    ε::T,
)::Nothing where {T<:half_quant}
    G, acc = size(norm, 3), zero(T)
    for g = 1:G
        acc = acc + norm[q, p, g]
    end
    log_Z[q, p] = log(acc + ε)
    return nothing
end

function log_norm(norm::AbstractArray{T,3}, ε::T)::AbstractArray{T,2} where {T<:half_quant}
    Q, P = size(norm, 1), size(norm, 2)
    log_Z = @zeros(Q, P)
    @parallel (1:Q, 1:P) log_norm_kernel!(log_Z, norm, ε)
    return log_Z
end

@parallel_indices (s) function mix_kernel!(
    logprob::AbstractArray{T,1},
    f::AbstractArray{T,3},
    α::AbstractArray{T,2},
    π_0::AbstractArray{T,3},
    Z::AbstractArray{T,2},
    ε::T,
    Q::Int,
    P::Int,
)::Nothing where {T<:half_quant}
    acc = zero(T)
    @inbounds for q = 1:Q
        acc_ = zero(T)
        @inbounds for p = 1:P
            acc_ = acc_ + (exp(f[q, p, s]) * π_0[q, 1, s] * α[q, p] / Z[q, p])
        end
        acc = acc + log(acc_ + ε)
    end
    logprob[s] = acc
    return nothing
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
    log_p = @zeros(S)
    @parallel (1:S) mix_kernel!(log_p, f, α, π_0, Z, ε, Q, P)
    return log_p
end

end
