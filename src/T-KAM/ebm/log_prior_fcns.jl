module LogPriorFCNs

export LogPriorULA, LogPriorMix, LogPriorUnivariate

using NNlib: softmax
using CUDA,
    KernelAbstractions,
    Tullio,
    Lux,
    LuxCUDA,
    LinearAlgebra,
    Accessors,
    Random,
    Tullio,
    ComponentArrays,
    ParallelStencil

using ..Utils
using ..EBM_Model

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, half_quant, 3)
else
    @init_parallel_stencil(Threads, half_quant, 3)
end

struct LogPriorULA{T<:half_quant} <: Lux.AbstractLuxLayer
    ε::T
end

struct LogPriorUnivariate{T<:half_quant} <: Lux.AbstractLuxLayer
    ε::T
    normalize::Bool
end

struct LogPriorMix{T<:half_quant} <: Lux.AbstractLuxLayer
    ε::T
    normalize::Bool
end

function (lp::LogPriorULA)(
    z::AbstractArray{T},
    ebm::EbmModel{T},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple,
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
    Q, P, S = size(z)
    log_π0 = @zeros(Q, P, S)
    ebm.π_pdf!(log_π0, z, ps.dist.π_μ, ps.dist.π_σ)
    @parallel (1:Q, 1:P, 1:S) stable_log!(log_π0, lp.ε)
    f, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, dropdims(z; dims = 2))
    return dropdims(sum(f; dims = 1) .+ log_π0; dims = 1), st_lyrnorm_new
end

@parallel_indices (q, p, s) function stable_log!(
    log_pdf::AbstractArray{T},
    ε::T,
)::Nothing where {T<:half_quant}
    log_pdf[q, p, s] = log(log_pdf[q, p, s] + ε)
    return nothing
end

@parallel_indices (q, p) function log_norm_kernel!(
    log_Z::AbstractArray{T},
    norm::AbstractArray{T},
    ε::T,
)::Nothing where {T<:half_quant}
    G, acc = size(norm, 3), zero(T)
    for g = 1:G
        acc = acc + norm[q, p, g]
    end
    log_Z[q, p] = log(acc + ε)
    return nothing
end

function (lp::LogPriorUnivariate)(
    z::AbstractArray{T},
    ebm,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple;
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
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
    log_π0 = @zeros(Q, P, S)
    ebm.π_pdf!(log_π0, z, ps.dist.π_μ, ps.dist.π_σ)
    @parallel (1:Q, 1:P, 1:S) stable_log!(log_π0, lp.ε)

    # Pre-allocate
    log_p = @zeros(S)
    log_Z = @zeros(Q, P)

    lp.normalize && @parallel (1:Q, 1:P) log_norm_kernel!(
        log_Z,
        first(ebm.quad(ebm, ps, st_kan, st_lyrnorm)),
        lp.ε,
    )

    for q = 1:Q
        f, st = ebm(ps, st_kan, st_lyrnorm, z[q, :, :])
        lp = f[q, :, :] .+ log_π0[q, :, :] .- log_Z[q, :]
        log_p = log_p + dropdims(sum(lp; dims = 1); dims = 1)
    end

    return log_p, st_lyrnorm
end

@parallel_indices (s) function mix_kernel!(
    logprob::AbstractArray{T},
    f::AbstractArray{T},
    log_απ::AbstractArray{T},
    log_Z::AbstractArray{T},
    reg::T,
    Q::Int,
    P::Int,
)::Nothing where {T<:half_quant}
    acc = zero(T)
    @inbounds for q = 1:Q
        @inbounds for p = 1:P
            acc = acc + log_απ[q, p, s] + f[q, p, s] - log_Z[q, p] + reg
        end
    end
    logprob[s] = acc
    return nothing
end

@parallel_indices (q, p, s) function stable_logalpha!(
    log_pdf::AbstractArray{T},
    alpha::AbstractArray{T},
    ε::T,
)::Nothing where {T<:half_quant}
    log_pdf[q, p, s] = log(log_pdf[q, 1, s] + alpha[q, p] + ε)
    return nothing
end

function (lp::LogPriorMix)(
    z::AbstractArray{T},
    ebm::EbmModel{T},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple,
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
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
    
    log_απ = @zeros(Q, P, S)
    ebm.π_pdf!(log_απ, z, ps.dist.π_μ, ps.dist.π_σ)
    @parallel (1:Q, 1:P, 1:S) stable_logalpha!(log_απ, alpha, lp.ε)

    # Energy functions of each component, q -> p
    f, st_lyrnorm = ebm(ps, st_kan, st_lyrnorm, dropdims(z; dims = 2))

    log_Z = @zeros(Q, P)
    lp.normalize && @parallel (1:Q, 1:P) log_norm_kernel!(
        log_Z,
        first(ebm.quad(ebm, ps, st_kan, st_lyrnorm)),
        lp.ε,
    )

    log_p = @zeros(S)
    @parallel (1:S) mix_kernel!(log_p, f, log_απ, log_Z, ebm.λ * sum(abs.(ps.dist.α)), Q, P)
    return log_p, st_lyrnorm
end

end
