module LogPriorFCNs

export prior_fwd, log_prior_univar, log_prior_ula, log_prior_mix

using CUDA, KernelAbstractions, Tullio, Lux, LuxCUDA, LinearAlgebra, Accessors, Random
using NNlib: softmax
using ChainRules: @ignore_derivatives

include("../../utils.jl")
include("../kan/univariate_functions.jl")
using .Utils: device, next_rng, half_quant, full_quant, fq
using .univariate_functions: fwd

function prior_fwd(
    ebm, 
    ps, 
    st, 
    z::AbstractArray{T}
    ) where {T<:half_quant}
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

    for i in 1:ebm.depth
        z = fwd(ebm.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = (i == 1 && !ebm.ula) ? reshape(z, size(z, 2), mid_size*size(z, 3)) : dropdims(sum(z, dims=1); dims=1)

        if ebm.layernorm && i < ebm.depth
            z, st_new = Lux.apply(ebm.fcns_qp[Symbol("ln_$i")], z, ps[Symbol("ln_$i")], st[Symbol("ln_$i")])
            @ignore_derivatives @reset st[Symbol("ln_$i")] = st_new
        end
    end

    z = ebm.ula ? z : reshape(z, ebm.q_size, ebm.p_size, :)
    return z, st
end

function log_prior_ula(
    ebm, 
    z::AbstractArray{T},
    ps, 
    st;
    ε::T=eps(half_quant),
    normalize::Bool=false
    ) where {T<:half_quant}
    log_π0 = ebm.prior_type == "learnable_gaussian" ? log.(ebm.π_pdf(z, ps, ε) .+ ε) : log.(ebm.π_pdf(z, ε) .+ ε)
    log_π0 = dropdims(sum(log_π0; dims=1);dims=1)
    f, st = prior_fwd(ebm, ps, st, dropdims(z; dims=2))
    return dropdims(sum(f; dims=1) .+ log_π0; dims=1), st
end

function log_prior_univar(
    ebm, 
    z::AbstractArray{T},
    ps, 
    st;
    ε::T=eps(half_quant),
    normalize::Bool=false
    ) where {T<:half_quant}
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

    log_π0 = ebm.prior_type == "learnable_gaussian" ? log.(ebm.π_pdf(z, ps, ε) .+ ε) : log.(ebm.π_pdf(z, ε) .+ ε)

    log_p = zeros(T, size(z)[end]) |> device
    log_Z = zeros(T, size(z)[1:2]...) |> device

    if normalize && !ebm.ula
        norm, _, st = ebm.quad(ebm, ps, st)
        log_Z = log.(dropdims(sum(norm; dims=3); dims=3) .+ ε)
    end

    for q in 1:size(z, 1)
        log_Zq = view(log_Z, q, :)
        f, st = prior_fwd(ebm, ps, st, z[q, :, :])
        lp = f[q, :, :] .+ log_π0[q, :, :]
        log_p += dropdims(sum(lp .- log_Zq; dims=1); dims=1)
    end
    return log_p, st
end

function log_prior_mix(
    ebm, 
    z::AbstractArray{T},
    ps, 
    st;
    ε::T=eps(half_quant),
    normalize::Bool=false,
    ) where {T<:half_quant}
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

    S = size(z,2)

    # Mixture proportions and prior
    alpha = softmax(ps[Symbol("α")]; dims=2) 
    π_0 = ebm.prior_type == "learnable_gaussian" ? ebm.π_pdf(z, ps, ε) : ebm.π_pdf(z, ε)
    log_απ = log.(reshape(alpha, size(alpha)..., 1) .* π_0 .+ ε)

    # Energy functions of each component, q -> p
    z, st = prior_fwd(ebm, ps, st, dropdims(z; dims=2))

    log_Z = zeros(T, size(z)) |> device
    if normalize
        norm, _, st = ebm.quad(ebm, ps, st)
        log_Z = reshape(log.(dropdims(sum(norm; dims=3); dims=3) .+ ε), ebm.q_size, 1, S) |> device
    end

    # Unnormalized or normalized log-probability
    logprob = z + log_απ
    logprob = logprob .- log_Z
    l1_reg = ebm.λ * sum(abs.(ps[Symbol("α")])) 
    return dropdims(sum(logprob; dims=(1,2)); dims=(1,2)) .+ l1_reg, st
end

end