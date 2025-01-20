module LangevinSampling

export MALA_sampler

using CUDA, KernelAbstractions, Tullio, LinearAlgebra, Random, Lux, LuxCUDA
using Zygote: gradient

include("mixture_prior.jl")
include("KAN_likelihood.jl")
include("../utils.jl")
using .ebm_mix_prior: log_prior
using .KAN_likelihood: log_likelihood
using .Utils: device, next_rng, quant

function MALA_sampler(
    m,
    ps,
    st,
    x::AbstractArray{quant};
    t::AbstractArray{quant}=device([quant(1)]),
    η::quant=quant(0.1),
    N::Int=20,
    burn_in::Int=0,
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
        η: The step size.
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

    # Pre-allocate buffers
    noise = similar(z)
    proposal = similar(z)
    grad_z = similar(z)
    grad_proposal = similar(z)

    # Avoid looped stochasticity
    seed, rng = next_rng(seed)
    noise = randn(quant, N, size(z)...) .* sqrt(2 * η) |> device
    seed, rng = next_rng(seed)
    log_u = log.(rand(rng, quant, N)) # Local proposals
    seed, rng = next_rng(seed)
    log_u_global = log.(rand(rng, quant, N)) # Global proposals
    seed, rng = next_rng(seed)
    swap_indices = T > 1 ? rand(rng, 2:T, N) : nothing 

    function log_posterior(z_i)
        lp = log_prior(m.prior, z_i, ps.ebm, st.ebm; normalize=false)'
        ll, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z_i; seed=seed, noise=false)
        lp, ll =reshape(lp, T, B, 1), reshape(ll, T, B, :)
        return sum(lp .+ (t .* ll))
    end

    # Local acceptance ratio within temperature
    function MH_local(proposal_i, z_i, grad_current, grad_proposal)

        # Posterior ratio
        log_acceptance_ratio = log_posterior(proposal_i) - log_posterior(z_i)

        # Transition kernels or drift corrections (gaussian)
        log_acceptance_ratio -= -sum((proposal_i - z_i - η * grad_current).^2) / 4η
        log_acceptance_ratio += -sum((z_i - proposal_i - η * grad_proposal).^2) / 4η
        
        return log_acceptance_ratio
    end

    # Global Replica Exchange accross temperatures
    function RE_global(z_low, z_high, t_low, t_high)
        ll_low, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z_low; seed=seed)
        ll_high, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z_high; seed=seed)
        ll_low, ll_high = sum(ll_low), sum(ll_high)

        # Log-acceptance is the likelihood of swap vs no swap
        log_acceptance_ratio = (t_high .* ll_low) + (t_low .* ll_high)
        log_acceptance_ratio -= (t_high .* ll_high) + (t_low .* ll_low)

        return log_acceptance_ratio
    end

    num_rejections = 0
    for i in 1:N
        grad_z .= first(gradient(z_i -> log_posterior(z_i), z))
        proposal .= z .+ (η .* grad_z) .+ (noise[i, :, :])
        grad_proposal .= first(gradient(z_i -> log_posterior(z_i), proposal))

        # Local Metropolis-Hastings acceptance
        log_α = MH_local(proposal, z, grad_z, grad_proposal)
        if log_u[i] < log_α || i < burn_in
            z .= proposal
        elseif i >= burn_in
            num_rejections += 1
        end

        # Global Replica Exchange
        if T > 1
            z = reshape(z, T, B, Q)
            for idx in 2:T
                z_low = z[idx-1, :, :]
                z_high = z[idx, :, :] 
                if log_u_global[i] < RE_global(z_low, z_high, view(t, idx-1), view(t, idx))
                    z[idx-1, :, :] .= z_high
                    z[idx, :, :] .= z_low
                end
            end
            z = reshape(z, T * B, Q)
        end
    end

    m.verbose && println("Rejection rate: ", num_rejections / N)

    z = reshape(z, T, B, Q)
    return z, seed
end

end




