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
    temperatures::AbstractArray{quant}=device([quant(1)]),
    η::quant=quant(0.1),
    σ::quant=quant(1),
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
        t: The temperatures if using Thermodynamic Integration.
        η: The step size.
        σ: The noise level, (leave at 1)
        N: The number of iterations.
        seed: The seed for the random number generator.

    Returns:
        The posterior samples.
        The updated seed.
    """
    # Initialize from prior
    z, seed = m.prior.sample_z(m.prior, m.MC_samples, ps.ebm, st.ebm, seed)
    T, B, Q = length(temperatures), size(z)...
    output = reshape(z, 1, B, Q) # Initialize output, accumulated over temperatures
    k = 1

    # Pre-allocate buffers
    noise = similar(z)
    proposal = similar(z)
    drift = similar(z)
    drift_proposal = similar(z)
    forward_drift = similar(z)
    backward_drift = similar(z)

    # Avoid looped stochasticity
    seed, rng = next_rng(seed)
    noise = randn(quant, N, size(z)...) .* σ .* sqrt(2 * η) |> device
    seed, rng = next_rng(seed)
    log_u = log.(rand(rng, quant, N)) # Local proposals
    seed, rng = next_rng(seed)

    function log_posterior(z_i, t_k)
        lp = log_prior(m.prior, z_i, ps.ebm, st.ebm; normalize=false)'
        ll, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z_i; seed=seed)
        return sum(lp .+ (t_k .* ll))
    end

    # Local acceptance ratio within temperature
    function MH_local(proposal_i, z_i, grad_current, grad_proposal, t_k)

        # Proposal densities (drift corrections)
        forward_drift .= proposal_i - z_i - η * grad_current
        backward_drift .= z_i - proposal_i - η * grad_proposal

        # Log-acceptance is the difference in log-posterior and drift corrections
        log_acceptance_ratio = log_posterior(proposal_i, t_k) - log_posterior(z_i, t_k)
        log_acceptance_ratio += -sum(forward_drift.^2) / (2 * σ^2)
        log_acceptance_ratio += sum(backward_drift.^2) / (2 * σ^2)

        return log_acceptance_ratio
    end

    while k < T + 1
        t_k = view(temperatures, k)
        for i in 1:N
            drift .= first(gradient(z_i -> log_posterior(z_i, t_k), z))
            proposal .= z .+ (η .* drift) .+ (noise[i, :, :])
            drift_proposal .= first(gradient(z_i -> log_posterior(z_i, t_k), proposal))

            # Local Metropolis-Hastings acceptance
            if log_u[i] < MH_local(proposal, z, drift, drift_proposal, t_k)
                z .= proposal
            end
        end
        output = vcat(output, reshape(z, 1, B, Q))
        k += 1 # Move onto next temperature, retaining updated sample as initial state
    end

    return reshape(output, T+1, B, Q), seed
end

end




