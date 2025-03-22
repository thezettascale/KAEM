module LangevinSampling

export autoMALA_sampler

using CUDA, KernelAbstractions, Tullio, LinearAlgebra, Random, Lux, LuxCUDA, Distributions, Accessors, Statistics
using Zygote: withgradient
using Flux: mse

include("../utils.jl")
include("EBM_prior.jl")
include("KAN_likelihood.jl")
using .Utils: device, next_rng, half_quant, full_quant, fq
using .ebm_ebm_prior: log_prior
using .KAN_likelihood: log_likelihood

function cross_entropy_sum(x::AbstractArray{half_quant}, y::AbstractArray{half_quant}; ε::half_quant=eps(half_quant))
    log_x = log.(x .+ ε)
    ll = sum(log_x .* y)
    return ll ./ size(x, 1)
end

function sample_momentum(z::AbstractArray{full_quant}; seed::Int=1)
    """
    Sample momentum for the autoMALA sampler, (with pre-conditioner).

    Args:
        z: The current position.
        seed: The seed for the random number generator.

    Returns:
        The momentum.
        The positive-definite mass matrix, (only the diagonals are returned for efficiency).
        The updated seed.
    """
    Q, P, B = size(z)
    Σ = diag(cov(cpu_device()(reshape(z, Q*P, B)')))
    
    # Pre-conditioner
    seed, rng = next_rng(seed)
    β = rand(rng, Truncated(Beta(1, 1), 0.5, 2/3)) |> full_quant
    Σ_AM = β .* sqrt.(1 ./ Σ) .+ (1 - β)

    # Momentum
    seed, rng = next_rng(seed)
    p = rand(rng, MvNormal(zeros(full_quant, length(Σ_AM)), Diagonal(Σ_AM)), B)

    p = reshape(p, Q, P, B)
    Σ_AM = reshape(Σ_AM, Q, P)
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
    logpos_withgrad::Function
    )
    """
    Generate a proposal for a particular step-size.

    Args:
        z: The current position.
        momentum: The current momentum.
        η: The step size.
        logpos: The log-posterior function.
        seed: The seed for the random number generator.

    Returns:
        The proposal.
        The log-ratio.
    """

    p = momentum .+ (η .* ∇z / 2) # Half-step momentum update
    ẑ = z .+ (η .* p) ./ M # Full-step position update

    logpos_ẑ, ∇ẑ, st = logpos_withgrad(ẑ, st)
    p = p + (η .* ∇ẑ / 2) # Half-step momentum update

    # MH acceptance ratio
    log_r = logpos_ẑ - logpos_z - ((sum(p.^2) - sum(momentum.^2)) / 2)
    return ẑ, logpos_ẑ, ∇ẑ, -p, log_r, st
end

function select_step_size(
    log_a::full_quant,
    log_b::full_quant,
    z::AbstractArray{full_quant},
    st,
    logpos_z::full_quant,
    ∇z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η_init::full_quant,
    Δη::full_quant,
    logpos_withgrad::Function
    )
    """
    Select a step size for the autoMALA sampler.

    Args:
        a: The lower bound.
        b: The upper bound.
        z: The current position.
        st: The state of the model.
        logpos_z: The log-posterior at the current position.
        momentum: The current momentum.
        M: The mass matrix.
        η_init: The initial step size.
        Δη: The step size increment.
        logpos_withgrad: The log-posterior function.

    Returns:
        The step size.
        The log-ratio.
        The updated state.
    """
    MH_criterion = (η) -> leapfrop_proposal(z, st, logpos_z, ∇z, momentum, M, η, logpos_withgrad)
    
    ẑ, logpos_ẑ, ∇ẑ, p̂, log_r, st = MH_criterion(η_init)
    if log_r <= log_a
        while log_r <= log_a
            η_init /= Δη
            ẑ, logpos_ẑ, ∇ẑ, p̂, log_r, st = MH_criterion(η_init)
        end
    elseif log_r >= log_b
        while log_r >= log_b
            η_init *= Δη
            ẑ, logpos_ẑ, ∇ẑ, p̂, log_r, st = MH_criterion(η_init)
        end
        η_init /= Δη
        ẑ, logpos_ẑ, ∇ẑ, p̂, log_r, st = MH_criterion(η_init)
    end
    return ẑ, logpos_ẑ, ∇ẑ, p̂, η_init, log_r, st
end

function autoMALA_step(
    log_a::full_quant,
    log_b::full_quant,
    z::AbstractArray{full_quant},
    st,
    logpos_z::full_quant,
    ∇z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η_init::full_quant,
    Δη::full_quant,
    logpos_withgrad::Function;
    )
    """
    Check the reversibility of the autoMALA step size selection.

    Args:
        a: The lower bound.
        b: The upper bound.
        z: The current position.
        st: The state of the model.
        logpos_z: The log-posterior at the current position.
        momentum: The current momentum.
        M: The mass matrix.
        η_init: The initial step size.
        Δη: The step size increment.
        logpos_withgrad: The log-posterior function.

    Returns:
        The step size.
        The log-ratio.
        The updated state.
    """
    ẑ, logpos_ẑ, ∇ẑ, p̂, η, log_r, st = select_step_size(log_a, log_b, z, st, logpos_z, ∇z, momentum, M, η_init, Δη, logpos_withgrad)
    _, _, _, _, η_prime, _, st = select_step_size(log_a, log_b, ẑ, st, logpos_ẑ, ∇ẑ, p̂, M, η, Δη, logpos_withgrad)
    return ẑ, η, η ≈ η_prime, log_r, st
end

function autoMALA_sampler(
    m,
    ps,
    st,
    x::AbstractArray{half_quant};
    t::AbstractArray{half_quant}=[half_quant(1)],
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
    z, st_ebm, seed = m.prior.sample_z(m.prior, m.IS_samples, ps.ebm, st.ebm, seed)
    @reset st.ebm = st_ebm
    z = z .|> full_quant
    loss_scaling = m.loss_scaling |> full_quant

    if isa(st.η_init, CuArray)
        @reset st.η_init = st.η_init |> cpu_device()
    end

    T, Q, P, B = length(t), size(z)...
    output = reshape(z, Q, P, B, 1)

    # Avoid looped stochasticity
    seed, rng = next_rng(seed)
    log_u = log.(rand(rng, full_quant, N, T))  
    seed, rng = next_rng(seed)
    ratio_bounds = log.(rand(rng, Uniform(0,1), N, T, 2)) .|> full_quant

    # Lkhood model based on type
    ll_fn = m.lkhood.seq_length > 1 ? (x,y) -> cross_entropy_sum(x, y; ε=m.ε) : (x,y) -> mse(x, y; agg=sum)

    function log_posterior(z_i::AbstractArray{half_quant}, st_i, t_k::half_quant)
        lp, st_ebm = log_prior(m.prior, z_i, ps.ebm, st_i.ebm; ε=m.ε)
        ll, st_gen, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z_i; seed=seed, ε=m.ε)
        logpos = sum(lp) + t_k * sum(ll)
        return logpos * m.loss_scaling, st_ebm, st_gen
    end

    k = 1
    num_acceptances = zeros(Int, T) 
    mean_η = zeros(full_quant, T) 
    while k < T + 1
        
        logpos_withgrad = (z_i, st_i) -> begin
            result = CUDA.@fastmath withgradient(z_j -> log_posterior(z_j, Lux.trainmode(st_i), t[k]), half_quant.(z_i))
            logpos_z, st_ebm, st_gen, ∇z = result.val..., first(result.grad)
            
            @reset st_i.ebm = st_ebm
            @reset st_i.gen = st_gen
            return full_quant(logpos_z) / loss_scaling, full_quant.(∇z) / loss_scaling, st_i
        end
        
        burn_in = 0
        η = st.η_init[k]
        for i in 1:N
            momentum, M, seed = sample_momentum(z; seed=seed) # Momentum
            log_a, log_b = min(ratio_bounds[i, k, :]...), max(ratio_bounds[i, k, :]...) # Bounds
            logpos_z, ∇z, st = logpos_withgrad(half_quant.(z), st) # Current position
            
            if burn_in < N_unadjusted
                z, logpos_ẑ, ∇ẑ, p̂, log_r, st = leapfrop_proposal(z, st, logpos_z, ∇z, momentum, M, η, logpos_withgrad) 
                burn_in += 1
            else
                ẑ, η, reversibility, log_r, st = autoMALA_step(log_a, log_b, z, st, logpos_z, ∇z, momentum, M, η, Δη, logpos_withgrad)
                if reversibility && (log_u[i, k] < log_r)
                    z .= ẑ
                    num_acceptances[k] += 1
                    mean_η[k] += η
                end
            end
        end
        output = cat(output, reshape(z, Q, P, B, 1); dims=4)
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
