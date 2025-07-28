module Losses

export IS_loss, MALA_loss

using ..Utils
using CUDA, ParallelStencil

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

## Fcns for model with Importance Sampling ##
@parallel_indices (b, s) function cross_entropy_IS!(
    ll::AbstractArray{T},
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
)::Nothing where {T<:half_quant}
    D, seq_length, acc = size(x)[1:2]..., zero(T)
    for d = 1:D, t = 1:seq_length
        acc = acc + log(x̂[d, t, s, b] + ε) * x[d, t, b]
    end
    ll[b, s] = acc / D / scale
    return nothing
end

@parallel_indices (b, s) function l2_IS!(
    ll::AbstractArray{T},
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
)::Nothing where {T<:half_quant}
    W, H, C, acc = size(x)[1:3]..., zero(T)
    for w = 1:W, h = 1:H, c = 1:C
        acc = acc + (x[w, h, c, b] - x̂[w, h, c, s, b]) ^ 2
    end
    ll[b, s] = - acc / scale
    return nothing
end

function IS_loss(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
    B::Int,
    S::Int,
    SEQ::Bool,
)::AbstractArray{T} where {T<:half_quant}
    ll = @zeros(B, S)
    stencil = SEQ ? cross_entropy_IS! : l2_IS!
    @parallel (1:B, 1:S) stencil(ll, x, x̂, ε, scale)
    return ll
end

## Fcns for model with Langevin methods ##
@parallel_indices (b) function cross_entropy_MALA!(
    ll::AbstractArray{T},
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
)::Nothing where {T<:half_quant}
    D, seq_length, acc = size(x)[1:2]..., zero(T)
    for d = 1:D, t = 1:seq_length
        acc = acc + log(x̂[d, t, b] + ε) * x[d, t, b]
    end
    ll[b] = acc / D / scale
    return nothing
end

@parallel_indices (b) function l2_MALA!(
    ll::AbstractArray{T},
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
)::Nothing where {T<:half_quant}
    W, H, C, acc = size(x)[1:3]..., zero(T)
    for w = 1:W, h = 1:H, c = 1:C
        acc = acc + (x[w, h, c, b] - x̂[w, h, c, b]) ^ 2
    end
    ll[b] = - acc / scale
    return nothing
end

function MALA_loss(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
    B::Int,
    SEQ::Bool,
)::AbstractArray{T} where {T<:half_quant}
    ll = @zeros(B)
    stencil = SEQ ? cross_entropy_MALA! : l2_MALA!
    @parallel (1:B) stencil(ll, x, x̂, ε, scale)
    return ll
end

end