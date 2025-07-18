module LogLikelihoods

export cross_entropy_IS,
    l2_IS, cross_entropy_MALA, l2_MALA, log_likelihood_IS, log_likelihood_MALA

using CUDA, KernelAbstractions, ComponentArrays, Random
using NNlib: softmax, sigmoid

include("../../utils.jl")
using .Utils: half_quant, full_quant, device

## Fcns for model with Importance Sampling ##
function cross_entropy_IS(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    noise::AbstractArray{T};
    ε::T = eps(T),
    act_fcn::Function = sigmoid,
)::AbstractArray{T} where {T<:half_quant}
    x̂ = act_fcn(permutedims(x̂ .+ noise, [1, 2, 4, 3]))
    log_x̂ = log.(x̂ .+ ε)
    ll = log_x̂ .* x
    ll = dropdims(sum(ll, dims = (1, 2)), dims = (1, 2)) # One-hot encoded cross-entropy
    return ll ./ size(x̂, 1)
end

function l2_IS(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    noise::AbstractArray{T};
    ε::T = eps(T),
    act_fcn::Function = sigmoid,
)::AbstractArray{T} where {T<:half_quant}
    x̂ = act_fcn(permutedims(x̂ .+ noise, [1, 2, 3, 5, 4]))
    ll = (x .- x̂) .^ 2
    return -dropdims(sum(ll, dims = (1, 2, 3)); dims = (1, 2, 3))
end

## Fcns for model with Lagenvin methods ##
function cross_entropy_MALA(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    noise::AbstractArray{T};
    ε::T = eps(half_quant),
    act_fcn::Function = sigmoid,
)::AbstractArray{T} where {T<:half_quant}
    x̂ = act_fcn(x̂ .+ noise)
    ll = log.(x̂ .+ ε) .* x ./ size(x, 1)
    return dropdims(sum(ll; dims = (1, 2)); dims = (1, 2))
end

function l2_MALA(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    noise::AbstractArray{T};
    ε::T = eps(half_quant),
    act_fcn::Function = sigmoid,
)::AbstractArray{T} where {T<:half_quant}
    x̂ = act_fcn(x̂ .+ noise)
    ll = (x - x̂) .^ 2
    return -dropdims(sum(ll; dims = (1, 2, 3)); dims = (1, 2, 3))
end

## Log-likelihood functions ##
function log_likelihood_IS(
    z::AbstractArray{T},
    x::AbstractArray{T},
    lkhood::Any,
    ps::ComponentArray{T},
    st::NamedTuple;
    noise::AbstractArray{T} = device(
        zeros(T, lkhood.x_shape..., size(z)[end], size(x)[end]),
    ),
    ε::T = eps(T),
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
    """
    Conditional likelihood of the generator.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        tempered: Whether to use tempered likelihood.
        rng: The random number generator.

    Returns:
        The unnormalized log-likelihood.
    """
    x̂, st = lkhood.generate_from_z(lkhood, ps, st, z)
    noise = lkhood.σ_llhood * noise
    ll =
        lkhood.seq_length > 1 ?
        cross_entropy_IS(x, x̂, noise; ε = ε, act_fcn = lkhood.output_activation) :
        l2_IS(x, x̂, noise; ε = ε)
    ll = ll ./ (2*lkhood.σ_llhood^2)
    return ll, st
end

function log_likelihood_MALA(
    z::AbstractArray{T},
    x::AbstractArray{T},
    lkhood::Any,
    ps::ComponentArray{T},
    st::NamedTuple;
    noise::AbstractArray{T} = device(zeros(T, size(x))),
    ε::T = eps(T),
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
    """
    Conditional likelihood of the generator sampled by Langevin.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        rng: The random number generator.

    Returns:
        The unnormalized log-likelihood.
    """
    x̂, st = lkhood.generate_from_z(lkhood, ps, st, z)
    noise = lkhood.σ_llhood * noise
    ll =
        lkhood.seq_length > 1 ?
        cross_entropy_MALA(x, x̂, noise; ε = ε, act_fcn = lkhood.output_activation) :
        l2_MALA(x, x̂, noise; ε = ε)
    ll = ll ./ (2*lkhood.σ_llhood^2)
    return ll, st
end

end
