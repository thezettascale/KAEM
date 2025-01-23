module LangevinSampling

export autoMALA_sampler

using CUDA, KernelAbstractions, Tullio, LinearAlgebra, Random, Lux, LuxCUDA, Distributions
using Zygote: withgradient

include("mixture_prior.jl")
include("KAN_likelihood.jl")
include("../utils.jl")
using .ebm_mix_prior: log_prior
using .KAN_likelihood: generate_from_z
using .Utils: device, next_rng, quant

function leapfrop_proposal(
    z::AbstractArray{quant},
    momentum::AbstractArray{quant},
    η::quant,
    logpos::Function;
    seed::Int=1
    )
    """
    Generate a proposal using the leapfrog method.

    Args:
        z: The current position.
        momentum: The current momentum.
        η: The step size.
        logpos: The log-posterior function.
        seed: The seed for the random number generator.

    Returns:
        The proposal.
        The log-ratio.
        The updated seed.
    """
    result = withgradient(z_i -> logpos(z_i, seed), z)
    logpos_z, seed, ∇z = result.val..., first(result.grad) 

    p = momentum + (η .* ∇z / 2) # Half-step momentum update
    ẑ = z + η .* p # Full-step position update

    result =  withgradient(z_i -> logpos(z_i, seed), ẑ)
    logpos_ẑ, seed, ∇ẑ = result.val..., first(result.grad)
    p = p + (η .* ∇ẑ / 2) # Half-step momentum update

    # MH acceptance ratio
    log_r = logpos_ẑ - logpos_z - ((sum(p.^2) - sum(momentum.^2)) / 2)

    return ẑ, log_r, seed
end

function reversibility_check(
    z::AbstractArray{quant},
    ẑ::AbstractArray{quant},
    η::quant;
    tol::quant=quant(1e-6),
    seed::Int=1
    )
    """
    Check if the leapfrog proposal is reversible.

    Args:
        z: The current position.
        ẑ: The proposed position.
        η: The step size.
        tol: The tolerance.

    Returns:
        A boolean indicating if the proposal is reversible.
        The updated seed.
    """
    result = withgradient(z_i -> logpos(z_i, seed), ẑ)
    _, seed, ∇ẑ = result.val..., first(result.grad)

    p_rev = ((ẑ - z) ./ η) - (η .* ∇ẑ / 2)
    z_rev, _, seed = leapfrop_proposal(ẑ, -p_rev, η, logpos; seed=seed)

    return norm(z_rev - z) < tol, seed
end

function autoMALA_sampler(
    m,
    ps,
    st,
    x::AbstractArray{quant};
    t::AbstractArray{quant}=[quant(1)],
    N::Int=20,
    N_unadjusted::Int=1,
    η_init::quant=quant(0.1),
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
        η_init: The initial step size.
        seed: The seed for the random number generator.

    Returns:
        The posterior samples.
        The updated seed.
    """
    # Initialize from prior
    z, seed = m.prior.sample_z(m.prior, size(x, 2), ps.ebm, st.ebm, seed)
    T, B, Q = length(t), size(z)...
    
    output = reshape(z, 1, B, Q)

    # Avoid looped stochasticity
    seed, rng = next_rng(seed)
    noise = device(randn(quant, N, T, size(z)...)) |> device
    seed, rng = next_rng(seed)
    log_u = log.(rand(rng, quant, N, T))  
    seed, rng = next_rng(seed)
    ratio_bounds = log.(rand(rng, Uniform(0,1), N, T, 2)) .|> quant

    function log_posterior(z_i, t_k, seed_i)
        lp = log_prior(m.prior, z_i, ps.ebm, st.ebm; normalize=false)'
        x̂, seed_i = generate_from_z(m.lkhood, ps.gen, st.gen, z_i; seed=seed_i)
        ll = m.lkhood.log_lkhood_model(x, x̂)
        return sum(lp .+ t_k .* ll), seed_i
    end

    k = 1
    num_acceptances = Dict("t_$i" => 0 for i in 1:T)
    while k < T + 1
        logpos = (z_i, seed_i) -> log_posterior(z_i, t[k], seed_i)
        burn_in = 0
        for i in 1:N
            η = η_init
            momentum = noise[i, k, :, :] .* sqrt(2 * η)
            log_a, log_b = ratio_bounds[i, k, :]

            proposal, log_r, seed = leapfrop_proposal(z, momentum, η, logpos; seed=seed)

            if burn_in < N_unadjusted
                z = proposal
                burn_in += 1
            else
                geq_bool = log_r >= log_b
                while !(log_a < log_r < log_b)
                    η = geq_bool ? η * 2 : η / 2
                    proposal, log_r, seed = leapfrop_proposal(z, momentum, η, logpos; seed=seed)
                end
                η = geq_bool ? η / 2 : η

                reversibility, seed = reversibility_check(z, proposal, η; seed=seed)
                if reversibility && (log_u[i, k] < log_r)
                    z = proposal
                    num_acceptances["t_$k"] += 1
                end
            end
        end
        output = vcat(output, reshape(z, 1, B, Q))
        k += 1
    end

    m.verbose && println("Acceptance rates: ", num_acceptances)

    return output, seed
end

end
