module LangevinSampling

export autoMALA_sampler

using CUDA, KernelAbstractions, Tullio, LinearAlgebra, Random, Lux, LuxCUDA, Distributions, Accessors, Statistics
using Zygote: gradient

include("../utils.jl")
using .Utils: device, next_rng, half_quant, full_quant

# Gaussian for testing purposes
if occursin("langevin_tests.jl", string(@__FILE__))
    log_prior(m, z, ps, st; normalize=false) = full_quant(0)
    log_likelihood(m, ps, st, x, z; seed=1) = -sum(full_quant(z).^2) ./ 2, st, seed
else
    include("mixture_prior.jl")
    include("KAN_likelihood.jl")
    using .ebm_mix_prior: log_prior
    using .KAN_likelihood: log_likelihood
end

function sample_momentum(z::AbstractArray{full_quant}; seed::Int=1)
    """
    Sample momentum for the autoMALA sampler, (with pre-conditioner).

    Args:
        z: The current position.
        seed: The seed for the random number generator.

    Returns:
        The momentum.
        The positive-definite mass matrix.
        The updated seed.
    """
    Σ, Q = cov(cpu_device()(z)), size(z, 2)
    
    # Pre-conditioner
    seed, rng = next_rng(seed)
    ε = rand(rng, Truncated(Beta(1, 1), 0.5, 2/3)) |> full_quant
    Σ_AM = zeros(Q, Q) 
    for i in 1:Q
        Σ_AM[i,i] = ε * sqrt(1/Σ[i,i]) + (1 - ε)
    end

    # Momentum
    seed, rng = next_rng(seed)
    p = rand(rng, MvNormal(zeros(size(Σ_AM, 1)), Σ_AM), size(z, 1))'

    # all(eigvals(Σ_AM) .> 0) && error("Mass matrix is not positive-definite.")

    return device(p), device(Σ_AM), seed
end

function leapfrop_proposal(
    z::AbstractArray{full_quant},
    st,
    logpos_z::full_quant,
    ∇z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η::full_quant,
    logpos::Function;
    uniform_prior::Bool=false,
    seed::Int=1
    )
    """
    Generate a proposal.

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
    p = momentum + (η .* ∇z / 2) # Half-step momentum update
    ẑ = z + (η .* p*M) # Full-step position update

    # Apply reflective boundary conditions, (which has a Jacobian of 1, so no need to adjust the log-ratio)
    if uniform_prior
        p = ifelse.(0 .< ẑ .< 1, p, -p) |> device
        ẑ = ifelse.(ẑ .< 0, -ẑ, ẑ) |> device
        ẑ = ifelse.(ẑ .> 1, 2 .- ẑ, ẑ) |> device
    end

    ∇ẑ = first(gradient(z_i -> first(logpos(z_i, st, false, seed)), half_quant.(ẑ))) .|> full_quant
    logpos_ẑ, st_new, seed = logpos(half_quant.(ẑ), Lux.testmode(st), true, seed)
    @reset st.gen = st_new

    p = p + (η .* ∇ẑ / 2) # Half-step momentum update

    # MH acceptance ratio
    log_r = logpos_ẑ - logpos_z - ((sum(p.^2) - sum(momentum.^2)) / 2)

    return ẑ, log_r, st, seed
end

function reversibility_check(
    z::AbstractArray{full_quant},
    st,
    ẑ::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η::full_quant,
    logpos::Function;
    tol::full_quant=full_quant(1e-4),
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
    ∇ẑ = first(gradient(z_i -> first(logpos(z_i, st, false, seed)), half_quant.(ẑ))) .|> full_quant
    logpos_∇ẑ, st_new, seed = logpos(half_quant.(ẑ), Lux.testmode(st), true, seed)
    @reset st.gen = st_new

    p_rev = M \ (((ẑ - z) ./ η) - (η .* ∇ẑ / 2))'
    z_rev, _, st, seed = leapfrop_proposal(ẑ, st, logpos_∇ẑ, ∇ẑ, -p_rev', M, η, logpos; seed=seed)

    return norm(z_rev - z) < tol, st, seed
end

function autoMALA_sampler(
    m,
    ps,
    st,
    x::AbstractArray{half_quant};
    t::AbstractArray{full_quant}=[full_quant(1)],
    N::Int=20,
    N_unadjusted::Int=1,
    Δη::full_quant=full_quant(2),
    η_min::full_quant=full_quant(1e-5),
    η_max::full_quant=full_quant(1),
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
    z, seed = m.prior.sample_z(m.prior, m.IS_samples, ps.ebm, st.ebm, seed)
    z = z .|> full_quant

    if isa(st.η_init, CuArray)
        @reset st.η_init = st.η_init |> cpu_device()
    end

    T, B, Q = length(t), size(z)...
    output = reshape(z, 1, B, Q)

    # Avoid looped stochasticity
    seed, rng = next_rng(seed)
    log_u = log.(rand(rng, full_quant, N, T))  
    seed, rng = next_rng(seed)
    ratio_bounds = log.(rand(rng, Uniform(0,1), N, T, 2)) .|> full_quant

    function log_posterior(z_i::AbstractArray{half_quant}, st_i, t_k::full_quant; full_precision::Bool=false, seed_i::Int=1)
        lp = log_prior(m.prior, z_i, ps.ebm, st_i.ebm; normalize=false, full_precision=full_precision)'
        ll, st_new, seed_i = log_likelihood(m.lkhood, ps.gen, st_i.gen, x, z_i; full_precision=full_precision, seed=seed_i)
        return sum(lp .+ t_k .* ll), st_new, seed_i
    end

    k = 1
    num_acceptances = zeros(Int, T) 
    mean_η = zeros(full_quant, T) 
    while k < T + 1
        logpos = (z_i, st_i, prec, seed_i) -> log_posterior(z_i, st_i, t[k]; full_precision=prec, seed_i=seed_i)
        burn_in = 0
        for i in 1:N
            η = st.η_init[k]
            momentum, M, seed = sample_momentum(z; seed=seed)
            log_a, log_b = min(ratio_bounds[i, k, :]...), max(ratio_bounds[i, k, :]...)

            ∇z = first(gradient(z_i -> first(logpos(z_i, st, false, seed)), half_quant.(z))) .|> full_quant
            logpos_z, st_gen, seed = logpos(half_quant.(z), Lux.testmode(st), true, seed)
            @reset st.gen = st_gen

            proposal, log_r, st, seed = leapfrop_proposal(z, st, logpos_z, ∇z, momentum, M, η, logpos; seed=seed)

            if burn_in < N_unadjusted
                z .= proposal
                burn_in += 1
            else
                geq_bool = log_r >= log_b
                while !(log_a < log_r < log_b) && (η_min <= η <= η_max)
                    η = geq_bool ? η * Δη : η / Δη
                    proposal, log_r, st, seed = leapfrop_proposal(z, st, logpos_z, ∇z, momentum, M, η, logpos; seed=seed)
                end
                η = geq_bool ? η / Δη : η

                reversibility, st, seed = reversibility_check(z, st, proposal, M, η, logpos; seed=seed)
                if reversibility && (log_u[i, k] < log_r)
                    z .= proposal
                    num_acceptances[k] += 1
                    mean_η[k] += η
                end
            end
        end
        output = vcat(output, reshape(z, 1, B, Q))
        k += 1
    end

            
    # Update step size for next training iteration
    mean_η = clamp.(mean_η ./ num_acceptances, η_min, η_max)
    for k in 1:T
        mean_η[k] = ifelse(isnan(mean_η[k]), st.η_init[k], mean_η[k])
    end
    @reset st.η_init .= mean_η

    m.verbose && println("Acceptance rates: ", num_acceptances ./ (N - N_unadjusted))
    m.verbose && println("Mean step sizes: ", mean_η)

    return half_quant.(output), st, seed
end

end
