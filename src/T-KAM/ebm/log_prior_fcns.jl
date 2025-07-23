module LogPriorFCNs

export prior_fwd, log_prior_univar, log_prior_ula, log_prior_mix

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

using NNlib: softmax

include("../../utils.jl")
include("../kan/univariate_functions.jl")
using .Utils: half_quant, full_quant, fq, symbol_map
using .UnivariateFunctions

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, half_quant, 3)
else
    @init_parallel_stencil(Threads, half_quant, 3)
end

function prior_fwd(
    ebm,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple,
    z::AbstractArray{T},
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
    """
    Forward pass through the ebm-prior, returning the energy function.

    Args:
        ebm: The ebm-prior.
        ps: The parameters of the ebm-prior.
        st: The states of the ebm-prior.
        z: The component-wise latent samples to evaulate the measure on, (q, num_samples) or (p, num_samples)

    Returns:
        f: The energy function, (num_samples,) or (q, p, num_samples)
        st: The updated states of the ebm-prior.
    """

    mid_size = !ebm.mixture_model ? ebm.p_size : ebm.q_size

    for i = 1:ebm.depth
        z = ebm.fcns_qp[i](z, ps.fcn[symbol_map[i]], st_kan[symbol_map[i]])

        z =
            (i == 1 && !ebm.ula) ? reshape(z, size(z, 2), mid_size*size(z, 3)) :
            dropdims(sum(z, dims = 1); dims = 1)

        z, st_lyrnorm_new =
            (ebm.layernorm_bool && i < ebm.depth) ?
            Lux.apply(ebm.layernorms[i], z, ps.layernorm[i], st_lyrnorm[i]) :
            (z, st_lyrnorm)

        (ebm.layernorm_bool && i < ebm.depth) && @reset st_lyrnorm[i] = st_lyrnorm_new
    end

    z = ebm.ula ? z : reshape(z, ebm.q_size, ebm.p_size, :)
    return z, st_lyrnorm
end

function log_prior_ula(
    z::AbstractArray{T},
    ebm,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple,
    ε::T = eps(half_quant),
    normalize::Bool = false,
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
    Q, P, S = size(z)
    log_π0 = @zeros(Q, P, S)
    @parallel (1:Q, 1:P, 1:S) ebm.π_pdf!(log_π0, z, ε, ps.dist.π_μ, ps.dist.π_σ)
    @. log_π0 = log(log_π0 + ε)
    log_π0 = dropdims(sum(log_π0; dims = 1); dims = 1)
    f, st_lyrnorm_new = prior_fwd(ebm, ps, st_kan, st_lyrnorm, dropdims(z; dims = 2))
    return dropdims(sum(f; dims = 1) .+ log_π0; dims = 1), st_lyrnorm_new
end

@parallel_indices (q, p, s) function stable_log!(
    log_pdf::AbstractArray{T},
    ε::T
)::Nothing where {T<:half_quant}
    log_pdf[q,p,s] = log(log_pdf[q,p,s] + ε)
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

function log_prior_univar(
    z::AbstractArray{T},
    ebm,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple;
    ε::T = eps(half_quant),
    normalize::Bool = false,
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
    @parallel (1:Q, 1:P, 1:S) ebm.π_pdf!(log_π0, z, ε, ps.dist.π_μ, ps.dist.π_σ)
    @parallel (1:Q, 1:P, 1:S) stable_log!(log_π0, ε)

    # Pre-allocate
    log_p = @zeros(S)
    log_Z = @zeros(Q, P)

    (normalize && !ebm.ula) && @parallel (1:Q, 1:P) log_norm_kernel!(
        log_Z,
        first(ebm.quad(ebm, ps, st_kan, st_lyrnorm)),
        ε,
    )

    for q in 1:Q
        f, st = prior_fwd(ebm, ps, st_kan, st_lyrnorm, z[q, :, :])
        lp = f[q, :, :] .+ log_π0[q, :, :] .- log_Z[q, :]
        log_p += dropdims(sum(lp; dims=1); dims=1)
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
    ε::T
)::Nothing where {T<:half_quant}
    log_pdf[q,p,s] = log(log_pdf[q,p,s] + alpha[q,p] + ε)
    return nothing
end

function log_prior_mix(
    z::AbstractArray{T},
    ebm,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple,
    ε::T = eps(half_quant),
    normalize::Bool = false,
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

    Q, P, S = size(z)

    # Mixture proportions and prior
    alpha = softmax(ps.dist.α; dims = 2)
    log_απ = @zeros(Q, P, S)
    @parallel (1:Q, 1:P, 1:S) ebm.π_pdf!(log_απ, z, ε, ps.dist.π_μ, ps.dist.π_σ)
    @parallel (1:Q, 1:P, 1:S) stable_logalpha!(log_απ, alpha, ε)

    # Energy functions of each component, q -> p
    f, st_lyrnorm = prior_fwd(ebm, ps, st_kan, st_lyrnorm, dropdims(z; dims = 2))

    log_Z = @zeros(Q, P)
    normalize && @parallel (1:Q, 1:P) log_norm_kernel!(
        log_Z,
        first(ebm.quad(ebm, ps, st_kan, st_lyrnorm)),
        ε,
    )

    log_p = @zeros(S)
    @parallel (1:S) mix_kernel!(log_p, f, log_απ, log_Z, ebm.λ * abs(ps.dist.α), Q, P)
    return log_p, st_lyrnorm
end

end
