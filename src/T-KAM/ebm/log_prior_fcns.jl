module LogPriorFCNs

export LogPriorULA, LogPriorMix, LogPriorUnivariate

using NNlib: logsoftmax, softmax
using CUDA, Lux, LuxCUDA, LinearAlgebra, Accessors, Random, ComponentArrays
using KernelAbstractions, Tullio

using ..Utils
using ..EBM_Model

function log_norm(norm::AbstractArray{T,3}, ε::T)::AbstractArray{T,2} where {T<:half_quant}
    return dropdims(log.(sum(norm, dims = 3) .+ ε), dims = 3)
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
    @tullio lp[q, s] := exp(f[q, p, s]) * π_0[q, 1, s] * α[q, p] / Z[q, p]
    lp = lp .+ ε
    @tullio out[s] := log(lp[q, s])
    return out
end

struct LogPriorULA{T<:half_quant} <: AbstractLogPrior
    ε::T
end

struct LogPriorUnivariate{T<:half_quant} <: AbstractLogPrior
    ε::T
    normalize::Bool
end

struct LogPriorMix{T<:half_quant} <: AbstractLogPrior
    ε::T
    normalize::Bool
end

function (lp::LogPriorULA)(
    z::AbstractArray{T,3},
    ebm::EbmModel{T},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple,
)::Tuple{AbstractArray{T,1},NamedTuple} where {T<:half_quant}
    Q, P, S = size(z)
    log_π0 = ebm.π_pdf(z, ps.dist.π_μ, ps.dist.π_σ; log_bool = true)
    f, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, dropdims(z; dims = 2))
    return dropdims(sum(f; dims = 1) .+ log_π0; dims = 1), st_lyrnorm_new
end

function (lp::LogPriorUnivariate)(
    z::AbstractArray{T,3},
    ebm,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple;
)::Tuple{AbstractArray{T,1},NamedTuple} where {T<:half_quant}
    """
    The log-probability of the ebm-prior.

    ∑_q [ ∑_p f_{q,p}(z_qp) ]

    Args:
        ebm: The ebm-prior.
        z: The component-wise latent samples to evaulate the measure on, (num_samples, q)
        ps: The parameters of the ebm-prior.
        st: The states of the ebm-prior.
        normalize: Whether to normalize the log-probability.
        ε: The small value to avoid log(0).
        agg: Whether to sum the log-probability over the samples.

    Returns:
        The unnormalized log-probability of the ebm-prior.
        The updated states of the ebm-prior.
    """

    Q, P, S = size(z)
    log_π0 = ebm.π_pdf(z, ps.dist.π_μ, ps.dist.π_σ; log_bool = true)

    # Pre-allocate
    log_p = zeros(T, S) |> pu
    log_Z =
        lp.normalize ? log_norm(first(ebm.quad(ebm, ps, st_kan, st_lyrnorm)), lp.ε) :
        zeros(T, Q, P) |> pu

    for q = 1:Q
        f, st = ebm(ps, st_kan, st_lyrnorm, z[q, :, :])
        lp = f[q, :, :] .+ log_π0[q, :, :] .- log_Z[q, :]
        log_p = log_p + dropdims(sum(lp; dims = 1); dims = 1)
    end

    return log_p, st_lyrnorm
end

function (lp::LogPriorMix)(
    z::AbstractArray{T,3},
    ebm::EbmModel{T},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple,
)::Tuple{AbstractArray{T,1},NamedTuple} where {T<:half_quant}
    """
    The log-probability of the mixture ebm-prior.

    ∑_q [ log ( ∑_p α_p exp(f_{q,p}(z_q)) π_0(z_q) ) ]


    Args:
        mix: The mixture ebm-prior.
        z: The component-wise latent samples to evaulate the measure on, (num_samples, q)
        ps: The parameters of the mixture ebm-prior.
        st: The states of the mixture ebm-prior.
        normalize: Whether to normalize the log-probability.
        ε: The small value to avoid log(0).

    Returns:
        The unnormalized log-probability of the mixture ebm-prior.
        The updated states of the mixture ebm-prior.
    """
    alpha = softmax(ps.dist.α; dims = 2)
    Q, P, S = size(alpha)..., size(z)[end]
    π_0 = ebm.π_pdf(z, ps.dist.π_μ, ps.dist.π_σ; log_bool = false)

    # Energy functions of each component, q -> p
    f, st_lyrnorm = ebm(ps, st_kan, st_lyrnorm, dropdims(z; dims = 2))
    Z =
        lp.normalize ? dropdims(sum(first(ebm.quad(ebm, ps, st_kan, st_lyrnorm)), dims = 3), dims = 3) :
        ones(T, Q, P) |> pu

    reg = ebm.λ > 0 ? ebm.λ * sum(abs.(alpha)) : zero(T)
    log_p = log_mix_pdf(f, alpha, π_0, Z, lp.ε, Q, P, S)
    return log_p .+ reg, st_lyrnorm
end

end
