module Losses

export IS_loss, MALA_loss

using ..Utils

## Fcns for model with Importance Sampling ##
function cross_entropy_IS(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
)::AbstractArray{T} where {T<:half_quant}
    log_x̂ = log.(x̂ .+ ε)
    ll_expanded = permutedims(log_x̂, [1, 2, 4, 3]) .* x
    ll = dropdims(sum(ll_expanded, dims = (1, 2)), dims = (1, 2)) # One-hot encoded cross-entropy
    return ll ./ size(x̂, 1) ./ scale
end

function l2_IS(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
)::AbstractArray{T} where {T<:half_quant}
    ll_expanded = (x .- permutedims(x̂, [1, 2, 3, 5, 4])) .^ 2
    ll = dropdims(sum(ll_expanded, dims = (1, 2, 3)); dims = (1, 2, 3))
    return - ll ./ scale
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
    ll = log.(x̂ .+ ε) .* x
    return dropdims(sum(ll; dims = (1, 2)); dims = (1, 2)) ./ size(x, 1) ./ scale
end

function l2_MALA(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
)::AbstractArray{T} where {T<:half_quant}
    ll = -(x - x̂) .^ 2
    return dropdims(sum(ll; dims = (1, 2, 3)); dims = (1, 2, 3)) ./ scale
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
