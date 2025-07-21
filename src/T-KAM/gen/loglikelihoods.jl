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
    x̂::AbstractArray{T};
    ε::T = eps(T),
) where {T<:half_quant}
    log_x̂ = log.(x̂ .+ ε)
    ll = permutedims(log_x̂, [1, 2, 4, 3]) .* x
    ll = dropdims(sum(ll, dims = (1, 2)), dims = (1, 2)) # One-hot encoded cross-entropy
    return ll ./ size(x̂, 1)
end

function l2_IS(
    x::AbstractArray{T},
    x̂::AbstractArray{T};
    ε::T = eps(T),
) where {T<:half_quant}
    ll = (x .- permutedims(x̂, [1, 2, 3, 5, 4])) .^ 2
    return -dropdims(sum(ll, dims = (1, 2, 3)); dims = (1, 2, 3))
end

## Fcns for model with Langevin methods
function cross_entropy_MALA(
    x::AbstractArray{T},
    x̂::AbstractArray{T};
    ε::T = eps(half_quant),
) where {T<:half_quant}
    ll = log.(x̂ .+ ε) .* x ./ size(x, 1)
    return dropdims(sum(ll; dims = (1, 2)); dims = (1, 2))
end

function l2_MALA(
    x::AbstractArray{T},
    x̂::AbstractArray{T};
    ε::T = eps(half_quant),
) where {T<:half_quant}
    ll = (x - x̂) .^ 2
    return -dropdims(sum(ll; dims = (1, 2, 3)); dims = (1, 2, 3))
end

## Log-likelihood functions ##
function log_likelihood_IS(
    z::AbstractArray{T},
    x::AbstractArray{T},
    lkhood,
    ps::ComponentArray{T},
    st::NamedTuple,
    noise::AbstractArray{T};
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
    x̂_noised = lkhood.output_activation(x̂ .+ noise)
    ll =
        lkhood.seq_length > 1 ? cross_entropy_IS(x, x̂_noised; ε = ε) :
        l2_IS(x, x̂_noised; ε = ε)
    return ll ./ (2*lkhood.σ_llhood^2), st
end

function log_likelihood_MALA(
    z::AbstractArray{T},
    x::AbstractArray{T},
    lkhood,
    ps::ComponentArray{T},
    st::NamedTuple;
    ε::T = eps(half_quant),
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
    x̂_act = lkhood.output_activation(x̂)
    ll =
        lkhood.seq_length > 1 ? cross_entropy_MALA(x, x̂_act; ε = ε) :
        l2_MALA(x, x̂_act; ε = ε)
    return ll ./ (2*lkhood.σ_llhood^2), st
end

end
