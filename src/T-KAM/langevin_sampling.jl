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
    η::quant=quant(0.1),
    σ::quant=quant(0.1),
    N::Int=20,
    seed::Int=1,
    )
    """
    Metropolis-adjusted Langevin algorithm (MALA) sampler to generate posterior samples.

    Args:
        m: The model.
        ps: The parameters of the model.
        st: The states of the model.
        x: The data
        η: The step size.
        σ: The noise level.
        N: The number of iterations.
        seed: The seed for the random number generator.

    Returns:
        The posterior samples.
        The updated seed.
    """
    # Initialize from prior
    z, seed = m.prior.sample_z(m.prior, m.MC_samples, ps.ebm, st.ebm, seed)

    function log_posterior(z_i)
        lp = log_prior(m.prior, z_i, ps.ebm, st.ebm)
        ll, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z_i)
        return sum(lp' .+ ll)
    end

    function MH_acceptance(proposal_i, z_i)
        grad_current = first(gradient(log_posterior, z_i))
        grad_proposal = first(gradient(log_posterior, proposal_i))

        # Proposal densities (drift corrections)
        forward_drift = proposal_i - z_i - η * grad_current
        backward_drift = z_i - proposal_i - η * grad_proposal

        # Log-acceptance is the difference in log-posterior and drift corrections
        log_acceptance_ratio = log_posterior(proposal_i) - log_posterior(z_i)
        log_acceptance_ratio += -sum(forward_drift.^2) / (2 * σ^2)
        log_acceptance_ratio += sum(backward_drift.^2) / (2 * σ^2)

        return log_acceptance_ratio
    end


    for i in 1:N
        drift = first(gradient(log_posterior, z))
        
        seed, rng = next_rng(seed)
        noise = randn(rng, quant, size(z)) .* sqrt(2 * η) |> device

        proposal = z .+ (η .* drift) .+ (σ * noise)
        log_α = MH_acceptance(proposal, z)

        seed, rng = next_rng(seed)
        u = rand(rng, quant) |> device

        if log(u) < log_α
            z = proposal
        end
    end

    return z, seed
end

end




