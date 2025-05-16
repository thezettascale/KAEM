module autoMALA_sampling

export autoMALA_sampler, cross_entropy, l2

using CUDA, KernelAbstractions, Tullio, LinearAlgebra, Random, Lux, LuxCUDA, Distributions, Accessors, Statistics
using Zygote: gradient

include("../../utils.jl")
include("preconditioner.jl")
include("hamiltonian.jl")
using .Utils: device, next_rng, half_quant, full_quant
using .Preconditioning
using .HamiltonianDynamics

function safe_step_size_update(
    η::AbstractArray{U}, 
    δ::AbstractArray{Int}, 
    Δη::U
    ) where {U<:full_quant}
    η_new = η .* Δη.^δ
    return ifelse.(isfinite.(η_new), η_new, η)
end

function check_reversibility(
    ẑ::AbstractArray{U}, 
    z::AbstractArray{U}, 
    η::AbstractArray{U}, 
    η_prime::AbstractArray{U};
    tol::U=full_quant(1e-6)
    ) where {U<:full_quant}
    # Both checks are required to maintain detailed balance
    # pos_diff = dropdims(maximum(abs.(ẑ - z); dims=(1,2)); dims=(1,2)) .< tol * maximum(abs.(z)) # leapfrog reversibility check
    step_diff = abs.(η - η_prime) .< tol .* η # autoMALA reversibility check
    return step_diff
end

function leapfrop_proposal(
    z::AbstractArray{U},
    x::AbstractArray{T},
    st,
    logpos_z::AbstractArray{U},
    ∇z::AbstractArray{U},
    momentum::AbstractArray{U},  # This is y = M^{-1/2}p
    M::AbstractArray{U},         # This is M^{1/2}
    η::AbstractArray{U},
    logpos_withgrad::Function,
    temps::AbstractArray{T}
    ) where {T<:half_quant, U<:full_quant}
    """
    Implements preconditioned Hamiltonian dynamics with transformed momentum:
    y*(x,y)   = y  + (eps/2)M^{-1/2}grad(log pi)(x)
    x'(x,y*)  = x  + eps M^{-1/2}y*
    y'(x',y*) = y* + (eps/2)M^{-1/2}grad(log pi)(x')
    """
    # # Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad) and full step position update
    p, ẑ = ndims(z) == 4 ? position_update_4d(z, momentum, ∇z, M, η) : position_update_3d(z, momentum, ∇z, M, η)

    # Get gradient at new position
    logpos_ẑ, ∇ẑ, st = logpos_withgrad(ẑ, x, st, temps)

    p = ndims(z) == 4 ? momentum_update_4d(p, ∇ẑ, M, η) : momentum_update_3d(p, ∇ẑ, M, η)

    # Hamiltonian difference for transformed momentum
    # H(x,y) = -log(pi(x)) + (1/2)||p||^2 since p ~ N(0,I)
    log_r = logpos_ẑ - logpos_z - dropdims(sum(p.^2; dims=(1,2)) - sum(momentum.^2; dims=(1,2)); dims=(1,2)) ./ 2

    return ẑ, logpos_ẑ, ∇ẑ, -p, log_r, st
end

function select_step_size(
    log_a::AbstractArray{U},
    log_b::AbstractArray{U},
    z::AbstractArray{U},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    st,
    logpos_z::AbstractArray{U},
    ∇z::AbstractArray{U},
    momentum::AbstractArray{U},
    M::AbstractArray{U},
    η_init::AbstractArray{U},
    Δη::U,
    logpos_withgrad::Function;
    η_min::U=full_quant(1e-5),
    η_max::U=one(full_quant),
    seq::Bool=false
    ) where {T<:half_quant, U<:full_quant}
    
    ẑ, logpos_ẑ, ∇ẑ, p̂, log_r, st = leapfrop_proposal(z, x, st, logpos_z, ∇z, momentum, M, η_init, logpos_withgrad, temps)

    δ = (log_r .>= log_b) - (log_r .<= log_a)
    active_chains = findall(δ .!= 0) |> cpu_device()
    isempty(active_chains) && return ẑ, logpos_ẑ, ∇ẑ, p̂, η_init, log_r, st
    
    geq_bool = log_r .>= log_b

    while !isempty(active_chains)

        η_init[active_chains] .= safe_step_size_update(
            η_init[active_chains],
            δ[active_chains],
            Δη
        )
        
        x_active = seq ? x[:,:,active_chains] : x[:,:,:,active_chains]
        ẑ_active, logpos_ẑ_active, ∇ẑ_active, p̂_active, log_r_active, st = 
            leapfrop_proposal(
                z[:,:,active_chains], 
                x_active,
                st, 
                logpos_z[active_chains], 
                ∇z[:,:,active_chains], 
                momentum[:,:,active_chains],
                M[:,:,active_chains],
                η_init[active_chains], 
                logpos_withgrad,
                temps[active_chains]
            )
        
        ẑ[:,:,active_chains] .= ẑ_active
        logpos_ẑ[active_chains] .= logpos_ẑ_active  
        ∇ẑ[:,:,active_chains] .= ∇ẑ_active
        p̂[:,:,active_chains] .= p̂_active
        log_r[active_chains] .= log_r_active

        δ[active_chains] .= ifelse.(δ[active_chains] .== 1 .&& log_r[active_chains] .< log_b[active_chains], 0, δ[active_chains])
        δ[active_chains] .= ifelse.(δ[active_chains] .== -1 .&& log_r[active_chains] .> log_a[active_chains], 0, δ[active_chains])
        δ[active_chains] .= ifelse.(isnan.(log_r[active_chains]), 0, δ[active_chains])
        δ[active_chains] .= ifelse.(η_min .< η_init[active_chains] .< η_max, δ[active_chains], 0)
        active_chains = findall(δ .!= 0) |> cpu_device()
    end

    # Reduce step size for chains that initially had too high acceptance with safety check
    η_init = safe_step_size_update(η_init, -1 .* geq_bool, Δη)
    return ẑ, logpos_ẑ, ∇ẑ, p̂, η_init, log_r, st
end

function autoMALA_step(
    log_a::AbstractArray{U},
    log_b::AbstractArray{U},
    z::AbstractArray{U},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    st,
    logpos_z::AbstractArray{U},
    ∇z::AbstractArray{U},
    momentum::AbstractArray{U},
    M::AbstractArray{U},
    η_init::AbstractArray{U},
    Δη::U,
    logpos_withgrad::Function;
    η_min::U=full_quant(1e-5),
    η_max::U=one(full_quant),
    ε::U=eps(full_quant),
    seq::Bool=false
    ) where {T<:half_quant, U<:full_quant}
    
    ẑ, logpos_ẑ, ∇ẑ, p̂, η, log_r, _ = select_step_size(
        log_a, log_b, z, x, temps, st, logpos_z, ∇z, momentum, M, η_init, Δη,
        logpos_withgrad; η_min=η_min, η_max=η_max, seq=seq
    )
    
    z_rev, _, _, _, η_prime, _, st = select_step_size(
        log_a, log_b, ẑ, x, temps, st, logpos_ẑ, ∇ẑ, p̂, M, η_init, Δη,
        logpos_withgrad; η_min=η_min, η_max=η_max, seq=seq
    )
    
    reversible = check_reversibility(z, z_rev, η, η_prime; tol=ε)
    return ẑ, η, η_prime, reversible, log_r, st
end

function autoMALA_sampler(
    m,
    ps,
    st,
    x::AbstractArray{T};
    temps::AbstractArray{T}=device([one(half_quant)]),
    N::Int=20,
    N_unadjusted::Int=1,
    Δη::U=full_quant(2),
    η_min::U=full_quant(1e-5),
    η_max::U=one(full_quant),
    RE_frequency::Int=10,
    seed::Int=1,
    ) where {T<:half_quant, U<:full_quant}  
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
    
    # Initialize from prior (already in bounded space)
    z, st_ebm, seed = m.prior.sample_z(m, size(x)[end]*length(temps), ps, st, seed)
    z = z .|> U
    loss_scaling = m.loss_scaling |> U

    T_length, Q, P, S = length(temps), size(z)[1:2]..., size(x)[end]
    z = reshape(z, Q, P, S, T_length)

    temps = repeat(reshape(temps, 1, T_length), S, 1)

    # Initialize preconditioner
    M = zeros(U, Q, P, 1, T_length) 
    z_cpu = cpu_device()(z)
    for k in 1:T_length
        M[:,:,1,k], seed = init_mass_matrix(view(z_cpu,:,:,:,k), seed)
    end
    @reset st.η_init = device(st.η_init)

    # Avoid looped stochasticity
    seed, rng = next_rng(seed)
    log_u = log.(rand(rng, U, S, T_length, N)) |> device
    seed, rng = next_rng(seed)
    ratio_bounds = log.(rand(rng, Uniform(0,1), S, T_length, 2, N)) .|> U |> device
    seed, rng = next_rng(seed)
    log_u_swap = log.(rand(rng, U, S, T_length, N)) |> device

    seq = m.lkhood.seq_length > 1
    x_t = seq ? repeat(x, 1, 1, 1, T_length) : repeat(x, 1, 1, 1, 1, T_length)
    
    log_llhood_fcn = (z_i, x_i, st_gen, t_i) -> begin
        x̂, st_gen = m.lkhood.generate_from_z(m.lkhood, ps.gen, st_gen, z_i)
        x̂ = m.lkhood.output_activation(x̂)
        return m.lkhood.MALA_ll_fcn(x_i, x̂; t=t_i, ε=m.ε, σ=m.lkhood.σ_llhood), st_gen
    end

    function log_posterior(z_i::AbstractArray{T}, x_i::AbstractArray{T}, st_i, t::AbstractArray{T}) 
        st_ebm, st_gen = st_i.ebm, st_i.gen
        if ndims(z_i) == 4
            logpos = zeros(T, S, 0) |> device
            for k in 1:T_length
                z_k= view(z_i,:,:,:,k)
                x_k = seq ? view(x_i,:,:,:,k) : view(x_i,:,:,:,:,k)
                logprior, st_ebm = m.prior.lp_fcn(m.prior, z_k, ps.ebm, st_ebm; ε=m.ε)
                logllhood, st_gen = log_llhood_fcn(z_k, x_k, st_gen, view(t,:,k))
                logpos = hcat(logpos, logprior + logllhood)
            end
            return logpos .* m.loss_scaling, st_ebm, st_gen
        else
            logprior, st_ebm = m.prior.lp_fcn(m.prior, z_i, ps.ebm, st_ebm; ε=m.ε)
            logllhood, st_gen = log_llhood_fcn(z_i, x_i, st_gen, t)
            return (logprior + logllhood) .* m.loss_scaling, st_ebm, st_gen
        end
    end

    num_acceptances = zeros(Int, S, T_length) |> device
    mean_η = zeros(U, S, T_length) |> device    
    momentum = similar(z) |> cpu_device()
    
    logpos_withgrad = (z_i, x_i, st_i, t_k) -> begin
        logpos_z, st_ebm, st_gen = CUDA.@fastmath log_posterior(T.(z_i), x_i, Lux.testmode(st_i), t_k)
        ∇z = CUDA.@fastmath first(gradient(z_j -> sum(first(log_posterior(z_j, x_i, Lux.testmode(st_i), t_k))), T.(z_i)))
        @reset st_i.ebm = st_ebm
        @reset st_i.gen = st_gen
        return U.(logpos_z) ./ loss_scaling, U.(∇z) ./ loss_scaling, st_i
    end 

    burn_in = 0
    η = st.η_init

    pos_before = CUDA.@fastmath first(log_posterior(T.(z), x_t, Lux.testmode(st), temps)) ./ loss_scaling
    for i in 1:N
        z_cpu = cpu_device()(z)
        for k in 1:T_length
            momentum[:,:,:,k], M[:,:,1,k], seed = sample_momentum(view(z_cpu,:,:,:,k), M[:,:,1,k]; seed=seed)
        end

        log_a, log_b = dropdims(minimum(ratio_bounds[:,:,:,i]; dims=3); dims=3), dropdims(maximum(ratio_bounds[:,:,:,i]; dims=3); dims=3)
        logpos_z, ∇z, st = logpos_withgrad(z, x_t, st, temps)

        if burn_in < N_unadjusted
            burn_in += 1
            z, logpos_ẑ, ∇ẑ, p̂, log_r, st = 
            leapfrop_proposal(
                z, 
                x_t, 
                st, 
                logpos_z, 
                device(∇z), 
                device(momentum), 
                device(repeat(M, 1, 1, S, 1)), 
                η, 
                logpos_withgrad, 
                temps
                )
        else
            ẑ, η_prop, η_prime, reversible, log_r, st = autoMALA_step(
                log_a, 
                log_b, 
                z, 
                x_t, 
                temps,
                st, 
                logpos_z, 
                ∇z, 
                device(momentum), 
                device(repeat(M, 1, 1, S, 1)), 
                η, 
                U(Δη), 
                logpos_withgrad; 
                η_min=η_min, 
                η_max=η_max, 
                ε=U(m.ε), 
                seq=seq)

            accept = (view(log_u,:,:,i) .< log_r) .* reversible
            z = ẑ .* reshape(accept, 1, 1, S, T_length) + z .* reshape(1 .- accept, 1, 1, S, T_length)
            mean_η .= mean_η .+ η_prop .* accept
            η .= η_prop .* accept .+ η .* (1 .- accept)
            num_acceptances .= num_acceptances .+ accept

            # Replica exchange Monte Carlo
            if i % RE_frequency == 0 && T_length > 1
                for t in 1:T_length-1

                    # Global swap criterion
                    z_hq = T.(z)
                    ll_t, st_gen = log_llhood_fcn(view(z_hq,:,:,:,t), x, st.gen, view(temps, T_length))
                    ll_t1, st_gen = log_llhood_fcn(view(z_hq,:,:,:,t+1), x, st_gen, view(temps, T_length))
                    log_swap_ratio = (view(temps,t+1) - view(temps,t)) .* (ll_t - ll_t1)
                    
                    swap = view(log_u_swap,:,t,i) .< log_swap_ratio
                    @reset st.gen = st_gen
                    
                    # Swap samples where accepted
                    z[:,:,:,t] .= z[:,:,:,t] .* reshape(swap, 1, 1, S) + z[:,:,:,t+1] .* reshape(1 .- swap, 1, 1, S)
                    z[:,:,:,t+1] .= z[:,:,:,t+1] .* reshape(swap, 1, 1, S) + z[:,:,:,t] .* reshape(1 .- swap, 1, 1, S)
                end
            end
        end


    end
    pos_after = CUDA.@fastmath first(log_posterior(T.(z), x_t, Lux.testmode(st), temps)) ./ loss_scaling
    m.verbose && println("Posterior change: $(dropdims(mean(pos_after - pos_before; dims=1); dims=1))")

    mean_η = clamp.(mean_η ./ num_acceptances, η_min, η_max)
    mean_η = ifelse.(isnan.(mean_η), st.η_init, mean_η) |> device
    @reset st.η_init = mean_η

    m.verbose && println("Acceptance rates: ", dropdims(mean(num_acceptances ./ (N - N_unadjusted); dims=1); dims=1))
    m.verbose && println("Mean step sizes: ", dropdims(mean(mean_η; dims=1); dims=1))
    
    any(isnan.(z)) && error("NaN in z") 
    return T.(z), st, seed
end
end