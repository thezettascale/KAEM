module LangevinSampling

export langevin_sampler, cross_entropy, l2

using CUDA, KernelAbstractions, Tullio, LinearAlgebra, Random, Lux, LuxCUDA, Distributions, Accessors, Statistics
using Zygote: gradient

include("../../utils.jl")
include("../EBM_prior.jl")
using .Utils: device, next_rng, half_quant, full_quant, fq
using .ebm_ebm_prior: log_prior

function cross_entropy(x::AbstractArray{half_quant}, y::AbstractArray{half_quant}; ε::half_quant=eps(half_quant))
    return log.(x .+ ε) .* y ./ size(x, 1)
end

function l2(x::AbstractArray{half_quant}, y::AbstractArray{half_quant}; ε::half_quant=eps(half_quant))
    return -(x - y).^2
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
    Q, P, S = size(z)
    z = cpu_device()(z)
    
    # Covariance across chains
    Σ = diag(cov(cpu_device()(reshape(z, Q*P, S))'))

    # Pre-conditioner
    seed, rng = next_rng(seed)
    β = rand(rng, Truncated(Beta(1, 1), 0.5, 2/3)) |> full_quant
    Σ_AM = (β .* sqrt.(1 ./ Σ) .+ (1 - β)) .^ 2

    # Momentum
    seed, rng = next_rng(seed)
    p = rand(rng, MvNormal(zeros(full_quant, length(Σ_AM)), Diagonal(Σ_AM)), S)
    return device(reshape(p, Q, P, S)), device(reshape(Σ_AM, Q, P)), seed
end

function leapfrop_proposal(
    z::AbstractArray{full_quant},
    x::AbstractArray{half_quant},
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
        x: The data.
        momentum: The current momentum.
        η: The step size per chain.
        logpos: The log-posterior function.
        seed: The seed for the random number generator.

    Returns:
        The proposal.
        The log-ratio.
    """

    @tullio p_in[q,p,s] := momentum[q,p,s] + (η[s] .* ∇z[q,p,s] / 2) # Half-step momentum update
    @tullio ẑ[q,p,s] := z[q,p,s] + (η[s] .* p_in[q,p,s]) ./ M[q,p] # Full-step position update    
    logpos_ẑ, ∇ẑ, st = logpos_withgrad(ẑ, x, st)    
    @tullio p_out[q,p,s] := p_in[q,p,s] + (η[s] .* ∇ẑ[q,p,s] / 2) # Half-step momentum update
    log_r = logpos_ẑ - logpos_z - (dropdims(sum(p_out.^2; dims=(1,2)) - sum(momentum.^2; dims=(1,2)); dims=(1,2)) ./ 2)
    return ẑ, logpos_ẑ, ∇ẑ, -p_out, log_r, st
end

function select_step_size(
    log_a::full_quant,
    log_b::full_quant,
    z::AbstractArray{full_quant},
    x::AbstractArray{half_quant},
    st,
    logpos_z::AbstractVector{full_quant},
    ∇z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η_init::AbstractVector{full_quant},
    Δη::full_quant,
    logpos_withgrad::Function;
    seq::Bool=false,
    η_min::full_quant=full_quant(1e-5),
    η_max::full_quant=full_quant(1)
    )
    """
    Select a step size for the autoMALA sampler.

    Args:
        a: The lower bound.
        b: The upper bound.
        z: The current position.
        x: The data.
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
    ẑ, logpos_ẑ, ∇ẑ, p̂, log_r, st = leapfrop_proposal(z, x, st, logpos_z, ∇z, momentum, M, η_init, logpos_withgrad)

    # Find chains that need step size adjustment
    δ = (log_r .>= log_b) - (log_r .<= log_a)
    active_chains = findall(δ .!= 0)
    isempty(active_chains) && return ẑ, logpos_ẑ, ∇ẑ, p̂, η_init, log_r, st
    
    # Store which chains had too high acceptance rate initially
    geq_bool = log_r .>= log_b

    while !isempty(active_chains)
        η_init[active_chains] .*= Δη.^δ[active_chains]
        
        # Run only on chains needing adjustment (efficient)
        x_active = seq ? view(x, :, :, active_chains) : view(x, :, :, :, active_chains)
        ẑ_active, logpos_ẑ_active, ∇ẑ_active, p̂_active, log_r_active, st = 
            leapfrop_proposal(view(z,:,:,active_chains), x_active, st, view(logpos_z,active_chains), 
                            view(∇z,:,:,active_chains), view(momentum,:,:,active_chains),
                            M, view(η_init,active_chains), logpos_withgrad)
        
        # Update active chain results
        ẑ[:,:,active_chains] = ẑ_active
        logpos_ẑ[active_chains] = logpos_ẑ_active  
        ∇ẑ[:,:,active_chains] = ∇ẑ_active
        p̂[:,:,active_chains] = p̂_active
        log_r[active_chains] = log_r_active

        # Update which chains still need adjustment
        δ[active_chains] = ifelse.(δ[active_chains] .== 1 .&& log_r[active_chains] .< log_b, 0, δ[active_chains])
        δ[active_chains] = ifelse.(δ[active_chains] .== -1 .&& log_r[active_chains] .> log_a, 0, δ[active_chains])
        δ[active_chains] = ifelse.(η_min .< η_init[active_chains] .< η_max, δ[active_chains], 0)
        active_chains = findall(δ .!= 0)
    end

    # Reduce step size for chains that initially had too high acceptance
    η_init = ifelse.(geq_bool, η_init ./ Δη, η_init)
    return ẑ, logpos_ẑ, ∇ẑ, p̂, η_init, log_r, st
end

function autoMALA_step(
    log_a::full_quant,
    log_b::full_quant,
    z::AbstractArray{full_quant},
    x::AbstractArray{half_quant},
    st,
    logpos_z::AbstractVector{full_quant},
    ∇z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η_init::AbstractVector{full_quant},
    Δη::full_quant,
    logpos_withgrad::Function;
    seq::Bool=false,
    η_min::full_quant=full_quant(1e-5),
    η_max::full_quant=full_quant(1)
    )
    """
    Check the reversibility of the autoMALA step size selection.

    Args:
        a: The lower bound.
        b: The upper bound.
        z: The current position.
        x: The data.
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
    ẑ, logpos_ẑ, ∇ẑ, p̂, η, log_r, _ = select_step_size(log_a, log_b, z, x, st, logpos_z, ∇z, momentum, M, η_init, Δη, logpos_withgrad; seq=seq, η_min=η_min, η_max=η_max)
    _, _, _, _, η_prime, _, st = select_step_size(log_a, log_b, ẑ, x, st, logpos_ẑ, ∇ẑ, p̂, M, η_init, Δη, logpos_withgrad; seq=seq, η_min=η_min, η_max=η_max)
    return ẑ, η, η_prime, cpu_device()(η .≈ η_prime), cpu_device()(log_r), st
end

function langevin_sampler(
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
    max_zero_accept_iters::Int=50
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
    z, st_ebm, seed = m.prior.sample_z(m.prior, size(x)[end], ps.ebm, st.ebm, seed)
    @reset st.ebm = st_ebm
    z = z .|> full_quant
    loss_scaling = m.loss_scaling |> full_quant

    if isa(st.η_init, CuArray)
        @reset st.η_init = st.η_init |> cpu_device()
    end

    if isa(st.zero_accept_counter, CuArray)
        @reset st.zero_accept_counter = st.zero_accept_counter |> cpu_device()
    end

    T, Q, P, S = length(t), size(z)...
    output = reshape(z, Q, P, S, 1)

    # Avoid looped stochasticity
    seed, rng = next_rng(seed)
    log_u = log.(rand(rng, full_quant, N, T, S))  
    seed, rng = next_rng(seed)
    ratio_bounds = log.(rand(rng, Uniform(0,1), N, T, 2)) .|> full_quant

    seq = m.lkhood.seq_length > 1
    ll_fn = seq ? (x_i, y_i) -> dropdims(sum(cross_entropy(x_i, y_i; ε=m.ε); dims=1); dims=1) : (x_i, y_i) -> dropdims(sum(l2(x_i, y_i; ε=m.ε); dims=(1,2,3)); dims=(1,2,3))

    function log_posterior(z_i::AbstractArray{half_quant}, x_i::AbstractArray{half_quant}, st_i, t_k::half_quant)
        lp, st_ebm = log_prior(m.prior, z_i, ps.ebm, st_i.ebm; ε=m.ε)
        x̂, st_gen = m.lkhood.generate_from_z(m.lkhood, ps.gen, st_i.gen, z_i)
        x̂ = m.lkhood.output_activation(x̂) 
        logpos = lp + t_k * ll_fn(x̂, x_i) / (2*m.lkhood.σ_llhood^2)
        return logpos .* m.loss_scaling, st_ebm, st_gen
    end

    k = 1
    num_acceptances = zeros(Int, T, S)
    mean_η = zeros(full_quant, T, S)

    while k < T + 1

        # Determine if this temperature should use only unadjusted steps
        use_only_unadjusted = st.zero_accept_counter[k] >= max_zero_accept_iters
        local_burn_in = use_only_unadjusted ? N : N_unadjusted

        logpos_withgrad = (z_i, x_i, st_i) -> begin
            logpos_z, st_ebm, st_gen = CUDA.@fastmath log_posterior(half_quant.(z_i), x_i, Lux.testmode(st_i), t[k])
            ∇z = CUDA.@fastmath first(gradient(z_j -> sum(first(log_posterior(z_j, x_i, Lux.testmode(st_i), t[k]))), half_quant.(z_i)))
            
            @reset st_i.ebm = st_ebm
            @reset st_i.gen = st_gen
            return full_quant.(logpos_z) ./ loss_scaling, full_quant.(∇z) ./ loss_scaling, st_i
        end
        
        burn_in = 0
        η = device(st.η_init[k, 1:S])
        m.verbose && println("t=$(t[k]) posterior before update: ", sum(first(log_posterior(half_quant.(z), x, Lux.testmode(st), t[k]))) ./ loss_scaling)
        
        for i in 1:N
            momentum, M, seed = sample_momentum(z; seed=seed)
            log_a, log_b = min(ratio_bounds[i, k, :]...), max(ratio_bounds[i, k, :]...)
            logpos_z, ∇z, st = logpos_withgrad(half_quant.(z), x, st)

            if burn_in < local_burn_in
                z, logpos_ẑ, ∇ẑ, p̂, log_r, st = leapfrop_proposal(z, x, st, logpos_z, ∇z, momentum, M, η, logpos_withgrad)
                burn_in += 1
            else
                ẑ, η, η_prime, reversible, log_r, st = autoMALA_step(log_a, log_b, z, x, st, logpos_z, ∇z, momentum, M, η, Δη, logpos_withgrad; seq=seq, η_min=η_min, η_max=η_max)
                accept = log_u[i,k,:] .< log_r
                η_cpu = cpu_device()(η)
                for s in 1:S
                    if reversible[s] && accept[s]
                        z[:,:,s] .= ẑ[:,:,s]
                        num_acceptances[k,s] += 1
                        mean_η[k,s] += η_cpu[s]
                    end
                end
                η = (η + η_prime) ./ 2
            end
        end

        if sum(num_acceptances[k,:]) == 0
            st.zero_accept_counter[k] += 1
        else
            st.zero_accept_counter[k] = 0
        end

        m.verbose && println("t=$(t[k]) posterior after update: ", sum(first(log_posterior(half_quant.(z), x, Lux.testmode(st), t[k]))) ./ loss_scaling)
        output = cat(output, reshape(z, Q, P, S, 1); dims=4)
        k += 1
    end

    # Update step size for next training iteration - now per chain
    for s in 1:S
        mean_η[:,s] = clamp.(mean_η[:,s] ./ num_acceptances[:,s], η_min, η_max)
        for k in 1:T
            mean_η[k,s] = ifelse(isnan(mean_η[k,s]), st.η_init[k,s], mean_η[k,s])
        end
    end
    @reset st.η_init .= mean_η

    m.verbose && println("Mean acceptance rates: ", mean(num_acceptances ./ (N - N_unadjusted), dims=2)[:,1])
    m.verbose && println("Mean step sizes: ", mean(mean_η, dims=2))

    return half_quant.(output), st, seed
end

end
