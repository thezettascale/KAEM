module LogPriorFCNs

export LogPriorULA, LogPriorMix, LogPriorUnivariate

using NNlib: softmax
using CUDA, Lux, LuxCUDA, LinearAlgebra, Accessors, Random, ComponentArrays

using ..Utils
using ..EBM_Model

if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    include("lp_utils_gpu.jl")
    using .LogPriorUtils
else
    include("lp_utils.jl")
    using .LogPriorUtils
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
    log_π0 = ebm.π_pdf(z, ps.dist.π_μ, ps.dist.π_σ; log_bool = true)
    f, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, dropdims(z; dims = 2))
    return dropdims(sum(f; dims = 1) .+ log_π0; dims = 1), st_lyrnorm_new
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

    log_απ = ebm.π_pdf(z, ps.dist.π_μ, ps.dist.π_σ; log_bool = false)
    log_απ = log_alpha(log_απ, alpha, lp.ε, Q, P, S)

    # Energy functions of each component, q -> p
    f, st_lyrnorm = ebm(ps, st_kan, st_lyrnorm, dropdims(z; dims = 2))
    log_Z =
        lp.normalize ? log_norm(first(ebm.quad(ebm, ps, st_kan, st_lyrnorm)), lp.ε) :
        zeros(T, Q, P) |> pu

    log_p = log_mix_pdf(f, log_απ, log_Z, ebm.λ * sum(abs.(ps.dist.α)), Q, P, S)
    return log_p, st_lyrnorm
end

end
