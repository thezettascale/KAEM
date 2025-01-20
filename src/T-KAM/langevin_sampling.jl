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
    η::quant=qugrad_z .= first(gradient(z_i -> log_posterior(z_i, t_k), z))ant(0.1),
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
        N: The number of iterations.
        seed: The seed for the random number generator.

    Returns:
        The posterior samples.
        The updated seed.
    """
    # Initialize from prior
    z, seed = m.prior.sample_z(m.prior, m.MC_samples, ps.ebm, st.ebm, seed)
    T, k = length(temperatures), 1

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

    function log_posterior(z_i, t_k)
        lp = log_prior(m.prior, z_i, ps.ebm, st.ebm; normalize=false)'
        ll, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z_i; seed=seed, noise=false)
        return sum(lp .+ (t_k .* ll))
    end

    # Local acceptance ratio within temperature
    function MH_local(proposal_i, z_i, grad_current, grad_proposal, t_k)

        # Posterior ratio
        log_acceptance_ratio = log_posterior(proposal_i, t_k) - log_posterior(z_i, t_k)

        # Transition kernels or drift corrections (gaussian)
        log_acceptance_ratio -= -norm(proposal_i - z_i - η * grad_current)^2 / 4η
        log_acceptance_ratio += -norm(z_i - proposal_i - η * grad_proposal)^2 / 4η
        
        return log_acceptance_ratio
    end

    num_rejections = Dict("t_$(i)" => 0 for i in 1:T)
    while k < T + 1
        t_k = view(temperatures, k)
        for i in 1:N
            grad_z .= first(gradient(z_i -> log_posterior(z_i, t_k), z))
            proposal .= z .+ (η .* grad_z) .+ (noise[i, :, :])
            grad_proposal .= first(gradient(z_i -> log_posterior(z_i, t_k), proposal))

            # Local Metropolis-Hastings acceptance
            if log_u[i] < MH_local(proposal, z, grad_z, grad_proposal, t_k)
                z .= proposal
            else
                num_rejections["t_$(k)"] += 1
            end
        end
        k += 1 # Move onto next temperature, retaining updated sample as initial state
    end

    m.verbose && println("Rejection rates: ", num_rejections)
    return z, seed
end

end




