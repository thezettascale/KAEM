module LangevinSampling

export autoMALA_sampler

using CUDA, KernelAbstractions, Tullio, LinearAlgebra, Random, Lux, LuxCUDA, Distributions, Accessors, Statistics
using Zygote: withgradient

include("../utils.jl")
include("EBM_prior.jl")
include("KAN_likelihood.jl")
using .Utils: device, next_rng, half_quant, full_quant, fq
using .ebm_ebm_prior: log_prior
using .KAN_likelihood: log_likelihood

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
    Q, P, S = size(z)
    z = cpu_device()(z)
    
    # Calculate covariance per chain
    Σ = zeros(full_quant, Q*P, S)
    for s in 1:S
        z_s = reshape(z[:,:,s], Q*P)
        Σ[:,s] = z_s .* z_s # Diagonal covariance only
    end
    
    # Pre-conditioner per chain
    seed, rng = next_rng(seed)
    β = rand(rng, Truncated(Beta(1, 1), 0.5, 2/3), S) .|> full_quant
    Σ_AM = zeros(full_quant, Q*P, S)
    for s in 1:S
        Σ_AM[:,s] = (β[s] .* sqrt.(1 ./ Σ[:,s]) .+ (1 - β[s])) .^ 2
    end

    # Momentum per chain
    seed, rng = next_rng(seed)
    p = zeros(full_quant, Q*P, S)
    for s in 1:S
        p[:,s] = rand(rng, MvNormal(zeros(full_quant, Q*P), Diagonal(Σ_AM[:,s])))
    end
    
    return device(reshape(p, Q, P, S)), device(reshape(Σ_AM, Q, P, S)), seed
end

function leapfrop_proposal(
    z::AbstractArray{full_quant},
    st,
    logpos_z::AbstractVector{full_quant},
    ∇z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η::AbstractVector{full_quant},
    logpos_withgrad::Function
    )
    """
    Generate a proposal for a particular step-size.

    Args:
        z: The current position.
        momentum: The current momentum.
        η: The step size per chain.
        logpos: The log-posterior function.
        seed: The seed for the random number generator.

    Returns:
        The proposal.
        The log-ratio.
    """
    Q, P, S = size(z)
    ẑ = similar(z)
    logpos_ẑ = zeros(full_quant, S)
    ∇ẑ = similar(z)
    p = similar(momentum)
    log_r = zeros(full_quant, S)

    # Process each chain independently
    for s in 1:S
        p_s = momentum[:,:,s] .+ (η[s] .* ∇z[:,:,s] / 2) # Half-step momentum update
        ẑ[:,:,s] = z[:,:,s] .+ (η[s] .* p_s) ./ M[:,:,s] # Full-step position update
        
        # Get gradients for this chain
        logpos_ẑ_s, ∇ẑ_s, st = logpos_withgrad(ẑ[:,:,s:s], st)
        ∇ẑ[:,:,s] = ∇ẑ_s[:,:,1]
        logpos_ẑ[s] = logpos_ẑ_s
        
        p_s = p_s + (η[s] .* ∇ẑ[:,:,s] / 2) # Half-step momentum update
        p[:,:,s] = -p_s
        
        # MH acceptance ratio per chain
        log_r[s] = logpos_ẑ[s] - logpos_z[s] - ((sum(p_s.^2) - sum(momentum[:,:,s].^2)) / 2)
    end
    
    return ẑ, logpos_ẑ, ∇ẑ, p, log_r, st
end

function select_step_size(
    log_a::full_quant,
    log_b::full_quant,
    z::AbstractArray{full_quant},
    st,
    logpos_z::AbstractVector{full_quant},
    ∇z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η_init::AbstractVector{full_quant},
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
        η_init: The initial step size per chain.
        Δη: The step size increment.
        logpos_withgrad: The log-posterior function.

    Returns:
        The step size.
        The log-ratio.
        The updated state.
    """
    MH_criterion = (η) -> leapfrop_proposal(z, st, logpos_z, ∇z, momentum, M, η, logpos_withgrad)
    ẑ, logpos_ẑ, ∇ẑ, p̂, log_r, st = MH_criterion(η_init)

    # Check acceptance per chain
    δ = zeros(Int, size(z,3))
    for s in 1:size(z,3)
        δ[s] = (log_r[s] >= log_b) - (log_r[s] <= log_a)
    end
    
    all(δ .== 0) && return ẑ, logpos_ẑ, ∇ẑ, p̂, η_init, log_r, st

    η = copy(η_init)
    while true
        for s in 1:size(z,3)
            η[s] *= Δη^δ[s] # Adjust step size per chain
        end
        
        ẑ, logpos_ẑ, ∇ẑ, p̂, log_r, st = MH_criterion(η)
        any(isnan.(log_r)) && error("NaN in acceptance ratio")
        
        finished = true
        for s in 1:size(z,3)
            if δ[s] == 1 && log_r[s] >= log_b
                finished = false
            elseif δ[s] == -1 && log_r[s] <= log_a
                finished = false
            end
        end
        
        if finished
            for s in 1:size(z,3)
                if δ[s] == 1
                    η[s] /= Δη
                end
            end
            return ẑ, logpos_ẑ, ∇ẑ, p̂, η, log_r, st
        end
    end
end

function autoMALA_step(
    log_a::full_quant,
    log_b::full_quant,
    z::AbstractArray{full_quant},
    st,
    logpos_z::AbstractVector{full_quant},
    ∇z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η_init::AbstractVector{full_quant},
    Δη::full_quant,
    logpos_withgrad::Function;
    eps::half_quant=eps(half_quant),
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
        η_init: The initial step size per chain.
        Δη: The step size increment.
        logpos_withgrad: The log-posterior function.

    Returns:
        The step size.
        The log-ratio.
        The updated state.
    """
    ẑ, logpos_ẑ, ∇ẑ, p̂, η, log_r, _ = select_step_size(log_a, log_b, z, st, logpos_z, ∇z, momentum, M, η_init, Δη, logpos_withgrad)
    _, _, _, _, η_prime, _, st = select_step_size(log_a, log_b, ẑ, st, logpos_ẑ, ∇ẑ, p̂, M, η_init, Δη, logpos_withgrad)
    
    reversible = zeros(Bool, size(z,3))
    for s in 1:size(z,3)
        reversible[s] = isapprox(η[s], η_prime[s]; atol=eps)
    end
    
    return ẑ, η, η_prime, reversible, log_r, st
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

    T, Q, P, S = length(t), size(z)...
    output = reshape(z, Q, P, S, 1)

    # Avoid looped stochasticity
    seed, rng = next_rng(seed)
    log_u = log.(rand(rng, full_quant, N, T, S))  
    seed, rng = next_rng(seed)
    ratio_bounds = log.(rand(rng, Uniform(0,1), N, T, 2)) .|> full_quant

    function log_posterior(z_i::AbstractArray{half_quant}, st_i, t_k::half_quant)
        lp, st_ebm = log_prior(m.prior, z_i, ps.ebm, st_i.ebm; ε=m.ε)
        ll, st_gen, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z_i; seed=seed, ε=m.ε)
        logpos = sum(lp) + t_k * sum(ll)
        return logpos * m.loss_scaling, st_ebm, st_gen
    end

    k = 1
    num_acceptances = zeros(Int, T, S) 
    mean_η = zeros(full_quant, T, S) 
    while k < T + 1
        
        logpos_withgrad = (z_i, st_i) -> begin
            result = CUDA.@fastmath withgradient(z_j -> log_posterior(z_j, Lux.trainmode(st_i), t[k]), half_quant.(z_i))
            logpos_z, st_ebm, st_gen, ∇z = result.val..., first(result.grad)
            
            @reset st_i.ebm = st_ebm
            @reset st_i.gen = st_gen
            return full_quant(logpos_z) / loss_scaling, full_quant.(∇z) / loss_scaling, st_i
        end
        
        burn_in = 0
        η = st.η_init[k, :] # Per-chain per-temp step sizes
        for i in 1:N
            momentum, M, seed = sample_momentum(z; seed=seed) # Momentum
            log_a, log_b = min(ratio_bounds[i, k, :]...), max(ratio_bounds[i, k, :]...) # Bounds
            _, ∇z, st = logpos_withgrad(half_quant.(z), st) # Current position

            logpos_z = zeros(full_quant, S)
            for s in 1:S
                logpos_z[s] = first(logpos_withgrad(half_quant.(z[:,:,s:s]), st))
            end

            if burn_in < N_unadjusted
                z, logpos_ẑ, ∇ẑ, p̂, log_r, st = leapfrop_proposal(z, st, logpos_z, ∇z, momentum, M, η, logpos_withgrad) 
                burn_in += 1
            else
                ẑ, η, η_prime, reversible, log_r, st = autoMALA_step(log_a, log_b, z, st, logpos_z, ∇z, momentum, M, η, Δη, logpos_withgrad; eps=m.ε)
                for s in 1:S
                    if reversible[s] && log_u[i,k,s] < log_r[s]
                        z[:,:,s] .= ẑ[:,:,s]
                        num_acceptances[k,s] += 1
                        mean_η[k,s] += η[s]
                    end
                end
                η = (η + η_prime) / 2
            end
        end
        output = cat(output, reshape(z, Q, P, S, 1); dims=4)
        k += 1
    end

    # Update step size for next training iteration - now per chain
    for s in 1:S
        mean_η[:,s] = clamp.(mean_η[:,s] ./ num_acceptances[:,s], η_min, η_max)
        for k in 1:T
            mean_η[k,s] = ifelse(isnan(mean_η[k,s]), st.η_init[k], mean_η[k,s])
        end
    end
    @reset st.η_init .= mean_η

    m.verbose && println("Mean acceptance rates: ", mean(num_acceptances ./ (N - N_unadjusted), dims=2)[:,1])
    m.verbose && println("Mean step sizes: ", mean_η)

    return half_quant.(output), st, seed
end

end
