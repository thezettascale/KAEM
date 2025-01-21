module LangevinSampling

export MALA_sampler

using CUDA, KernelAbstractions, Tullio, LinearAlgebra, Random, Lux, LuxCUDA, Accessors
using Zygote: withgradient

include("mixture_prior.jl")
include("KAN_likelihood.jl")
include("../utils.jl")
using .ebm_mix_prior: log_prior
using .KAN_likelihood: log_likelihood
using .Utils: device, next_rng, quant

function MH_local(
    z_i::AbstractArray{quant}, 
    proposal_i::AbstractArray{quant},
    logpos_z::quant,
    logpos_proposal::quant, 
    ∇z_i::AbstractArray{quant},
    ∇proposal_i::AbstractArray{quant};
    η::quant=quant(0.1)
    )
    """
    Returns the local Metropolis-Hastings acceptance ratio.
    i.e. [ posterior(z') x drift_correction(z, z') ] / [ posterior(z) x drift_correction(z', z) ]

    Args:
        z_i: The current state.
        proposal_i: The proposed state.
        logpos_z: The log-posterior of the current state.
        logpos_proposal: The log-posterior of the proposed state.
        ∇z_i: The gradient of the current state.
        ∇proposal_i: The gradient of the proposed state.
        η: The step size.

    Returns:
        The log-acceptance ratio.
    """

    # Posterior ratio and transition kernels/drift corrections, (gaussian)
    log_acceptance_ratio = logpos_proposal - logpos_z
    log_acceptance_ratio -= -sum((proposal_i - z_i - η * ∇z_i).^2) / 4η
    log_acceptance_ratio += -sum((z_i - proposal_i - η * ∇proposal_i).^2) / 4η
    return log_acceptance_ratio
end

function RE_global(
    z_low::AbstractArray{quant}, 
    z_high::AbstractArray{quant},
    t_low::AbstractArray{quant},
    t_high::AbstractArray{quant},
    ll_fcn::Function;
    seed::Int=1
    )
    """
    Returns the global Replica Exchange acceptance ratio.
    i.e., the tempered likelihoods of swap vs no swap

    Args:
        z_low: The current state.
        z_high: The proposed state.
        t_low: The temperature of the current state.
        t_high: The temperature of the proposed state.
        ll_fcn: The log-likelihood function.

    Returns:
        The log-acceptance ratio.
    """

    ll_low, seed = ll_fcn(z_low, seed)
    ll_high, seed = ll_fcn(z_high, seed)

    log_acceptance_ratio = (t_high .* ll_low) + (t_low .* ll_high)
    log_acceptance_ratio -= (t_high .* ll_high) + (t_low .* ll_low)

    return log_acceptance_ratio, seed
end

function MALA_step(
    z::AbstractArray{quant},
    noise::AbstractArray{quant},
    log_u_accept::quant,
    logpos::Function;
    η::quant=quant(0.1),
    seed::Int=1
)
    """
    MALA drift step. 

    Args:
        z: The current state.
        noise: The noise for the proposal.
        η: The step size.
        log_u_accept: The log of the acceptance threshold.
        logpos: The log-posterior function.

    Returns:
        The updated state.
        The updated step size.
    """

    # Current state
    result = withgradient(z_i -> logpos(z_i, seed), z)
    logpos_z, seed, ∇z = result.val..., first(result.grad)

    # Initial proposal
    proposal = z + (η .* ∇z) + (noise .* sqrt(2 * η))
    result = withgradient(z_i -> logpos(z_i, seed), proposal)
    logpos_proposal, seed, ∇proposal = result.val..., first(result.grad)

    # Acceptance ratio
    log_r = MH_local(z, proposal, logpos_z, logpos_proposal, ∇z, ∇proposal; η=η)
    return log_u_accept < log_r ? (proposal, seed) : (z, seed)
end

function ReplicaExchange(
    z::AbstractArray{quant},
    log_u_accept::AbstractArray{quant},
    temperatures::AbstractArray{quant},
    T::Int,
    B::Int,
    Q::Int,
    ll_fcn::Function;
    seed::Int=1
    )
    """
    Replica Exchange Monte Carlo, global swaps

    Args:
        z: The current state.
        log_u_accept: The log of the acceptance threshold.
        temperatures: Power posterior annealing parameter
        T: Number of temperatures
        B: MC sample size
        Q: z dimension

    Returns:
        The updated state.
    """
    z = reshape(z, T, B, Q)

    for k in 1:T-1
        z_low = z[k, :, :]
        z_high = z[k+1, :, :] 
        log_r, seed = RE_global(z_low, z_high, view(temperatures, k), view(temperatures, k+1), ll_fcn; seed=seed)
        if log_u_accept[k] < log_r
            z[k, :, :] .= z_high
            z[k+1, :, :] .= z_low
        end
    end
    return reshape(z, T * B, Q), seed
end

function MALA_sampler(
    m,
    ps,
    st,
    x::AbstractArray{quant};
    t::AbstractArray{quant}=device([quant(1)]),
    N::Int=20,
    N_unadjusted::Int=0,
    rejection_tol::quant=quant(0.1),
    Δη::quant=quant(0.1),
    minmax_η::Tuple{quant, quant}=(quant(0.001), quant(1)),
    seed::Int=1,
    )
    """
    Metropolis-adjusted Langevin algorithm (MALA) sampler to generate posterior samples.

    Args:
        m: The model.
        ps: The parameters of the model.
        st: The states of the model.
        x: The data
        t: The temperatures if using Thermodynamic Integration.
        N: The number of iterations.
        seed: The seed for the random number generator.

    Returns:
        The posterior samples.
        The updated seed.
    """
    # Initialize from prior
    z, seed = m.prior.sample_z(m.prior, m.MC_samples, ps.ebm, st.ebm, seed)
    T, B, Q = length(t), size(z)...
    z = repeat(reshape(z, 1, B, Q), T, 1, 1) 
    z = reshape(z, T*B, Q)

    # Avoid looped stochasticity
    seed, rng = next_rng(seed)
    noise = randn(quant, N, size(z)...) |> device
    seed, rng = next_rng(seed)
    log_u_local = log.(rand(rng, quant, N)) # Local proposals
    seed, rng = next_rng(seed)
    log_u_global = log.(rand(rng, quant, N, T-1)) # Global proposals

    function log_posterior(z_i, seed_i)
        lp = log_prior(m.prior, z_i, ps.ebm, st.ebm; normalize=false)'
        ll, seed_i = log_likelihood(m.lkhood, ps.gen, st.gen, x, z_i; seed=seed_i, noise=false)
        lp, ll = reshape(lp, T, B, 1), reshape(ll, T, B, :)
        ll = maximum(ll, dims=3) # For single sample in batch!
        return sum(lp .+ t .* ll), seed_i
    end

    function log_lkhood(z_i, seed_i)
        ll, seed_i = log_likelihood(m.lkhood, ps.gen, st.gen, x, z_i; seed=seed_i, noise=false) 
        ll = maximum(ll; dims=1) # For single sample in batch!
        return sum(ll), seed_i
    end

    local_step = (z, noise, log_u, seed_i) -> MALA_step(z, noise, log_u, log_posterior; η=st.η, seed=seed_i)
    global_swap = (z, log_u, seed_i) -> ReplicaExchange(z, log_u, t, T, B, Q, log_lkhood; seed=seed_i)
    
    num_rejections = 0
    for i in 1:N
        
        z_reverse = z
        
        # Local Metropolis-Hastings, (after burn-in)
        if i > N_unadjusted
            z, seed = local_step(z, noise[i, :, :], log_u_local[i], seed)
            num_rejections = z == z_reverse ? num_rejections + 1 : num_rejections
        else
            result = withgradient(z_i -> log_posterior(z_i, seed), z)
            _, seed, ∇z = result.val..., first(result.grad)
            z .= z + (st.η .* ∇z) + (noise[i, :, :] .* sqrt(2 * st.η))
        end

        # Global Replica Exchange
        z, seed = T > 1 ? global_swap(z, log_u_global[i, :], seed) : (z, seed)
    end

    rejection_rate = num_rejections / (N - N_unadjusted)
    if rejection_rate > 0.426 + rejection_tol
        @reset st.η = max((1-Δη) * st.η, minmax_η[1])
    elseif rejection_rate < 0.426 - rejection_tol
        @reset st.η = min((1+Δη) * st.η, minmax_η[2])
    end

    m.verbose && println("Rejection rate: ", rejection_rate, ", η: ", st.η)

    z = reshape(z, T, B, Q)
    return z, seed, st
end

end




