module LogPriorFCNs

export LogPriorULA, LogPriorMix, LogPriorUnivariate

using NNlib: logsoftmax, softmax, logsumexp
using LinearAlgebra, Accessors, ComponentArrays, Lux

using ..Utils
using ..EBM_Model

function log_norm(norm, ε)
    norm = sum(norm; dims = 3)
    log_norm = log.(norm .+ ε)
    return dropdims(log_norm, dims = 3)
end

function log_mix_pdf(
        f,
        α,
        π_0,
        Z,
        ε,
    )
    log_terms = f .+ log.(π_0 .* α .+ ε) .- log.(Z .+ ε)
    summed_p = logsumexp(log_terms; dims = 2)
    prod_q = sum(summed_p; dims = 1)
    return dropdims(prod_q; dims = (1, 2))
end

struct LogPriorULA{T <: Float32} <: AbstractLogPrior
    ε::T
end

struct LogPriorUnivariate{T <: Float32} <: AbstractLogPrior
    ε::T
    normalize::Bool
end

struct LogPriorMix{T <: Float32} <: AbstractLogPrior
    ε::T
    normalize::Bool
end

function (lp::LogPriorULA)(
        z,
        ebm,
        ps,
        st_kan,
        st_lyrnorm,
    )
    log_π0 = dropdims(
        sum(ebm.π_pdf(z, ps.dist.π_μ, ps.dist.π_σ; log_bool = true); dims = 1),
        dims = (1, 2),
    )
    f, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, dropdims(z; dims = 2))
    return dropdims(sum(f; dims = 1); dims = 1) + log_π0, st_lyrnorm_new
end

function reduce_q(i, z, ps, st_kan, st_lyrnorm, ebm)
    f, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, z[i, :, :])
    return f[i, :, :], st_lyrnorm_new
end

function (lp::LogPriorUnivariate)(
        z,
        ebm,
        ps,
        st_kan,
        st_lyrnorm,
        st_quad;
        ula = false,
        component_mask = nothing,
    )
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
    Q, P, S = ebm.q_size, ebm.p_size, ebm.s_size
    log_π0 = ebm.π_pdf(z, ps.dist.π_μ, ps.dist.π_σ; log_bool = true)

    log_π0 =
        lp.normalize && !ula ?
        log_π0 .- log_norm(
            first(
                ebm.quad(
                    ebm,
                    ps,
                    st_kan,
                    st_lyrnorm,
                    st_quad
                )
            ), lp.ε
        ) : log_π0

    st_lyrnorm_new = st_lyrnorm

    state = (1, zero(z[1, 1, :]))
    while first(state) <= Q
        i, log_p = state
        f_q, st_lyrnorm_new = reduce_q(
            i,
            z,
            ps,
            st_kan,
            st_lyrnorm_new,
            ebm
        )
        new_lp = log_p + dropdims(
            sum(
                f_q + log_π0[i, :, :];
                dims = 1
            ); dims = 1
        )
        state = (i + 1, new_lp)
    end

    log_p = last(state)
    return log_p, st_lyrnorm_new
end

function dotprod_attn(
        Q,
        K,
        z,
        s_size,
    )
    scale = sqrt(Float32(s_size))
    return dropdims(sum(Q .* z .* K .* z; dims = 3); dims = 3) ./ scale
end

function (lp::LogPriorMix)(
        z,
        ebm,
        ps,
        st_kan,
        st_lyrnorm,
        st_quad;
        ula = false,
        component_mask = nothing,
    )
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
    Q, P, S = ebm.q_size, ebm.p_size, ebm.s_size
    alpha =
        ebm.bool_config.use_attention_kernel ?
        dotprod_attn(ps.attention.Q, ps.attention.K, z, S) :
        (ebm.bool_config.train_props ? ps.dist.α : zero(ps.dist.α) .+ 1.0f0)

    alpha = softmax(alpha; dims = 2)
    π_0 = ebm.π_pdf(z, ps.dist.π_μ, ps.dist.π_σ; log_bool = false)

    # Energy functions of each component, q -> p
    f, st_lyrnorm = ebm(ps, st_kan, st_lyrnorm, dropdims(z; dims = 2))
    Z = PermutedDimsArray(
        sum(
            first(
                ebm.quad(
                    ebm,
                    ps,
                    st_kan,
                    st_lyrnorm,
                    st_quad;
                    component_mask = component_mask,
                    mix_bool = true
                )
            ), dims = 3
        ), (1, 3, 2)
    )

    reg = (
        ebm.λ > 0 ?
            ebm.λ * sum(abs.(alpha)) :
            0.0f0
    )

    log_p = log_mix_pdf(f, alpha, π_0, Z, lp.ε)
    return log_p .- reg, st_lyrnorm
end

end
