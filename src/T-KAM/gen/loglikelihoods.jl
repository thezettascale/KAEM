module LogLikelihoods

export log_likelihood_IS, log_likelihood_MALA

using CUDA, ComponentArrays, Random
using NNlib: softmax, sigmoid

using ..Utils
using ..T_KAM_model: GenModel

include("losses.jl")
using .Losses

## Log-likelihood functions ##
function log_likelihood_IS(
    z::AbstractArray{T,3},
    x::AbstractArray{T},
    lkhood::GenModel{T},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    noise::AbstractArray{T};
    ε::T = eps(T),
)::Tuple{AbstractArray{T,2},NamedTuple} where {T<:half_quant}
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
    B, S = size(x)[end], size(z)[end]
    x̂, st_lux = lkhood.generator(ps, st_kan, st_lux, z)
    noise = lkhood.σ.noise .* noise
    x̂_noised = lkhood.output_activation(x̂ .+ noise)

    ll = IS_loss(x, x̂_noised, ε, 2*lkhood.σ.llhood^2, B, S, lkhood.SEQ)
    return ll, st_lux
end

function log_likelihood_MALA(
    z::AbstractArray{T,3},
    x::AbstractArray{T},
    lkhood::GenModel{T},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    noise::AbstractArray{T};
    ε::T = eps(half_quant),
)::Tuple{AbstractArray{T,1},NamedTuple} where {T<:half_quant}
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
    B = size(z)[end]
    x̂, st_lux = lkhood.generator(ps, st_kan, st_lux, z)
    noise = lkhood.σ.noise .* noise
    x̂_act = lkhood.output_activation(x̂ .+ noise)

    ll = MALA_loss(x, x̂_act, ε, 2*lkhood.σ.llhood^2, B, lkhood.SEQ)
    return ll, st_lux
end

end
