module LogLikelihoods

export cross_entropy_IS,
    l2_IS, cross_entropy_MALA, l2_MALA, log_likelihood_IS, log_likelihood_MALA

using CUDA, KernelAbstractions, ComponentArrays, Random, ParallelStencil
using NNlib: softmax, sigmoid

include("../../utils.jl")
using .Utils: half_quant, full_quant

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
    B, S = size(x)[end], size(z)[end]
    x̂, st = lkhood.generate_from_z(lkhood, ps, st, z)
    noise = lkhood.σ_llhood * noise
    x̂_noised = lkhood.output_activation(x̂ .+ noise)

    ll = @zeros(B, S)
    stencil = lkhood.seq_length > 1 ? cross_entropy_IS! : l2_IS!
    @parallel (1:B, 1:S) stencil(ll, x, x̂_noised, ε, 2*lkhood.σ_llhood^2)
    return ll, st
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
    B = size(x)[end]
    x̂, st = lkhood.generate_from_z(lkhood, ps, st, z)
    x̂_act = lkhood.output_activation(x̂)

    ll = @zeros(B)
    stencil = lkhood.seq_length > 1 ? cross_entropy_MALA! : l2_MALA!
    @parallel (1:B) stencil(ll, x, x̂_act, ε, 2*lkhood.σ_llhood^2)
    return ll, st
end

end
