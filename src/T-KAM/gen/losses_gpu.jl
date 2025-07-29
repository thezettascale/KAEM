module Losses

export IS_loss, MALA_loss

using ..Utils

using CUDA, KernelAbstractions, Tullio

## Fcns for model with Importance Sampling ##
function cross_entropy_IS(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
)::AbstractArray{T} where {T<:half_quant}
    @tullio ce[d, t, s, b] := log(x̂[d, t, s, b] + ε) * x[d, t, b]
    return @tullio ll[b, s] := ce[d, t, s, b] / size(x̂, 1) / scale
end

function l2_IS(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
)::AbstractArray{T} where {T<:half_quant}
    @tullio l2[w, h, c, s, b] := (x[w, h, c, b] - x̂[w, h, c, s, b])^2
    return @tullio ll[b, s] := - l2[w, h, c, s, b] / scale
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
    loss_fcn = SEQ ? cross_entropy_IS : l2_IS
    return loss_fcn(x, x̂, ε, scale)
end

## Fcns for model with Langevin methods ##
function cross_entropy_MALA(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
)::AbstractArray{T} where {T<:half_quant}
    @tullio ce[d, t, b] := log(x̂[d, t, b] + ε) * x[d, t, b]
    return @tullio ll[b] := ce[d, t, b] / size(x, 1) / scale
end

function l2_MALA(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
)::AbstractArray{T} where {T<:half_quant}
    @tullio l2[w, h, c, b] := (x[w, h, c, b] - x̂[w, h, c, b])^2
    return @tullio ll[b] := - l2[w, h, c, b] / scale
end

function MALA_loss(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
    B::Int,
    SEQ::Bool,
)::AbstractArray{T} where {T<:half_quant}
    loss_fcn = SEQ ? cross_entropy_MALA : l2_MALA
    return loss_fcn(x, x̂, ε, scale)
end

end
