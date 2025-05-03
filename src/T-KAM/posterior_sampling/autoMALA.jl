module LangevinSampling

export langevin_sampler, cross_entropy, l2

using CUDA, KernelAbstractions, Tullio, LinearAlgebra, Random, Lux, LuxCUDA, Distributions, Accessors, Statistics
using Zygote: gradient

include("../../utils.jl")
include("../EBM_prior.jl")
include("preconditioner.jl")
include("hamiltonian.jl")
using .Utils: device, next_rng, half_quant, full_quant
using .ebm_ebm_prior: log_prior
using .Preconditioning
using .HamiltonianDynamics


function cross_entropy(x::AbstractArray{half_quant}, y::AbstractArray{half_quant}; ε::half_quant=eps(half_quant))
    return log.(x .+ ε) .* y ./ size(x, 1)
end

function l2(x::AbstractArray{half_quant}, y::AbstractArray{half_quant}; ε::half_quant=eps(half_quant))
    return -(x - y).^2
end

function safe_step_size_update(
    η::AbstractArray{full_quant}, 
    δ::AbstractArray{Int}, 
    Δη::full_quant
    )
    η_new = η .* Δη.^δ
    return ifelse.(isfinite.(η_new), η_new, η)
end

function check_reversibility(
    ẑ::AbstractArray{full_quant}, 
    z::AbstractArray{full_quant}, 
    η::AbstractArray{full_quant}, 
    η_prime::AbstractArray{full_quant};
    tol::full_quant=full_quant(1e-6)
    )
    # Both checks are required to maintain detailed balance
    # pos_diff = dropdims(maximum(abs.(ẑ - z); dims=(1,2)); dims=(1,2)) .< tol * maximum(abs.(z)) # leapfrog reversibility check
    step_diff = abs.(η - η_prime) .< tol .* η # autoMALA reversibility check
    return step_diff
end

function leapfrop_proposal(
    z::AbstractArray{full_quant},
    x::AbstractArray{half_quant},
    st,
    logpos_z::AbstractArray{full_quant},
    ∇z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},  # This is y = M^{-1/2}p
    M::AbstractArray{full_quant},         # This is M^{1/2}
    η::AbstractArray{full_quant},
    logpos_withgrad::Function,
    domain::Tuple{full_quant, full_quant},
    temps::AbstractArray{half_quant}
    )
    """
    Implements preconditioned Hamiltonian dynamics with transformed momentum:
    y*(x,y)   = y  + (eps/2)M^{-1/2}grad(log pi)(x)
    x'(x,y*)  = x  + eps M^{-1/2}y*
    y'(x',y*) = y* + (eps/2)M^{-1/2}grad(log pi)(x')
    """
    # # Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad) and full step position update
    p, ẑ = ndims(z) == 4 ? position_update_4d(z, momentum, ∇z, M, η) : position_update_3d(z, momentum, ∇z, M, η)

    # Reflect at boundaries, both position and momentum
    reflect_low = ẑ .< first(domain)
    reflect_high = ẑ .> last(domain)
    ẑ = ifelse.(reflect_low, 2*first(domain) .- ẑ, ẑ)
    ẑ = ifelse.(reflect_high, 2*last(domain) .- ẑ, ẑ)
    p = ifelse.(reflect_low, -p, p)
    p = ifelse.(reflect_high, -p, p)

    # Get gradient at new position
    logpos_ẑ, ∇ẑ, st = logpos_withgrad(ẑ, x, st, temps)

    p = ndims(z) == 4 ? momentum_update_4d(p, ∇ẑ, M, η) : momentum_update_3d(p, ∇ẑ, M, η)

    # Hamiltonian difference for transformed momentum
    # H(x,y) = -log(pi(x)) + (1/2)||p||^2 since p ~ N(0,I)
    log_r = logpos_ẑ - logpos_z - dropdims(sum(p.^2; dims=(1,2)) - sum(momentum.^2; dims=(1,2)); dims=(1,2)) ./ 2

    return ẑ, logpos_ẑ, ∇ẑ, -p, log_r, st
end

function select_step_size(
    log_a::AbstractArray{full_quant},
    log_b::AbstractArray{full_quant},
    z::AbstractArray{full_quant},
    x::AbstractArray{half_quant},
    temps::AbstractArray{half_quant},
    st,
    logpos_z::AbstractArray{full_quant},
    ∇z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η_init::AbstractArray{full_quant},
    Δη::full_quant,
    domain::Tuple{full_quant, full_quant},
    logpos_withgrad::Function;
    η_min::full_quant=full_quant(1e-5),
    η_max::full_quant=full_quant(1),
    seq::Bool=false
    )
    
    ẑ, logpos_ẑ, ∇ẑ, p̂, log_r, st = leapfrop_proposal(z, x, st, logpos_z, ∇z, momentum, M, η_init, logpos_withgrad, domain, temps)

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
                domain,
                temps[active_chains]
            )
        
        ẑ[:,:,active_chains] .= ẑ_active
        logpos_ẑ[active_chains] .= logpos_ẑ_active  
        ∇ẑ[:,:,active_chains] .= ∇ẑ_active
        p̂[:,:,active_chains] .= p̂_active
        log_r[active_chains] .= log_r_active

        δ[active_chains] .= ifelse.(δ[active_chains] .== 1 .&& log_r[active_chains] .< log_b[active_chains], 0, δ[active_chains])
        δ[active_chains] .= ifelse.(δ[active_chains] .== -1 .&& log_r[active_chains] .> log_a[active_chains], 0, δ[active_chains])
        δ[active_chains] .= ifelse.(η_min .< η_init[active_chains] .< η_max, δ[active_chains], 0)
        active_chains = findall(δ .!= 0) |> cpu_device()
    end

    # Reduce step size for chains that initially had too high acceptance with safety check
    η_init = safe_step_size_update(η_init, -1 .* geq_bool, Δη)
    return ẑ, logpos_ẑ, ∇ẑ, p̂, η_init, log_r, st
end

function autoMALA_step(
    log_a::AbstractArray{full_quant},
    log_b::AbstractArray{full_quant},
    z::AbstractArray{full_quant},
    x::AbstractArray{half_quant},
    temps::AbstractArray{half_quant},
    st,
    logpos_z::AbstractArray{full_quant},
    ∇z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η_init::AbstractArray{full_quant},
    Δη::full_quant,
    domain::Tuple{full_quant, full_quant},
    logpos_withgrad::Function;
    η_min::full_quant=full_quant(1e-5),
    η_max::full_quant=full_quant(1),
    ε::full_quant=eps(full_quant),
    seq::Bool=false
    )
    
    ẑ, logpos_ẑ, ∇ẑ, p̂, η, log_r, _ = select_step_size(
        log_a, log_b, z, x, temps, st, logpos_z, ∇z, momentum, M, η_init, Δη, domain,
        logpos_withgrad; η_min=η_min, η_max=η_max, seq=seq
    )
    
    z_rev, _, _, _, η_prime, _, st = select_step_size(
        log_a, log_b, ẑ, x, temps, st, logpos_ẑ, ∇ẑ, p̂, M, η_init, Δη, domain,
        logpos_withgrad; η_min=η_min, η_max=η_max, seq=seq
    )
    
    reversible = check_reversibility(z, z_rev, η, η_prime; tol=ε)
    return ẑ, η, η_prime, reversible, log_r, st
end

function langevin_sampler(
    m,
    ps,
    st,
    x::AbstractArray{half_quant};
    temps::AbstractArray{half_quant}=half_quant.([0,1]),
    N::Int=20,
    N_unadjusted::Int=1,
    Δη::full_quant=full_quant(2),
    η_min::full_quant=full_quant(1e-5),
    η_max::full_quant=full_quant(1),
    RE_frequency::Int=10,
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
    # Get domain bounds
    domain = full_quant.(m.prior.fcns_qp[Symbol("1")].grid_range)
    
    # Initialize from prior (already in bounded space)
    z, st_ebm, seed = m.prior.sample_z(m.prior, size(x)[end]*length(temps), ps.ebm, st.ebm, seed)
    z = z .|> full_quant
    loss_scaling = m.loss_scaling |> full_quant

    T, Q, P, S = length(temps), size(z)[1:2]..., size(x)[end]
    z = reshape(z, Q, P, S, T)

    temps = repeat(reshape(temps, 1, T), S, 1)

    # Initialize preconditioner
    M = zeros(full_quant, Q, P, 1, T) 
    z_cpu = cpu_device()(z)
    for k in 1:T
        M[:,:,1,k], seed = init_mass_matrix(view(z_cpu,:,:,:,k), seed)
    end
    @reset st.η_init = device(st.η_init)

    # Avoid looped stochasticity
    seed, rng = next_rng(seed)
    log_u = log.(rand(rng, full_quant, S, T, N)) |> device
    seed, rng = next_rng(seed)
    ratio_bounds = log.(rand(rng, Uniform(0,1), S, T, 2, N)) .|> full_quant |> device
    seed, rng = next_rng(seed)
    log_u_swap = log.(rand(rng, full_quant, S, T, N)) |> device

    seq = m.lkhood.seq_length > 1
    ll_fn = seq ? (x_i, y_i) -> dropdims(sum(cross_entropy(x_i, y_i; ε=m.ε); dims=(1,2)); dims=(1,2)) : (x_i, y_i) -> dropdims(sum(l2(x_i, y_i; ε=m.ε); dims=(1,2,3)); dims=(1,2,3))
    x_t = seq ? repeat(x, 1, 1, 1, T) : repeat(x, 1, 1, 1, 1, T)
    
    log_llhood_fcn = (z_i, x_i, st_gen) -> begin
        x̂, st_gen = m.lkhood.generate_from_z(m.lkhood, ps.gen, st_gen, z_i)
        x̂ = m.lkhood.output_activation(x̂)
        return ll_fn(x̂, x_i) ./ (2*m.lkhood.σ_llhood^2), st_gen
    end

    function log_posterior(z_i::AbstractArray{half_quant}, x_i::AbstractArray{half_quant}, st_i, t_k::AbstractArray{half_quant})
        if ndims(z_i) == 4
            z_3D = reshape(z_i, Q, P, S*T)
            x_3D = seq ? reshape(x_i, size(x,1), size(x,2), S*T) : reshape(x_i, size(x,1), size(x,2), size(x,3), S*T)
            logprior, st_ebm = log_prior(m.prior, z_3D, ps.ebm, st_i.ebm; ε=m.ε)
            logllhood, st_gen = log_llhood_fcn(z_3D, x_3D, st_i.gen)
            logpos = reshape(logprior, S, T) + t_k .* reshape(logllhood, S, T)
            return logpos .* m.loss_scaling, st_ebm, st_gen
        else
            logprior, st_ebm = log_prior(m.prior, z_i, ps.ebm, st_i.ebm; ε=m.ε)
            logllhood, st_gen = log_llhood_fcn(z_i, x_i, st_i.gen)
            return (logprior + t_k .* logllhood) .* m.loss_scaling, st_ebm, st_gen
        end
    end

    k = 1
    num_acceptances = zeros(Int, S, T) |> device
    mean_η = zeros(full_quant, S, T) |> device
    momentum = similar(z) |> cpu_device()
    
    logpos_withgrad = (z_i, x_i, st_i, t_k) -> begin
        logpos_z, st_ebm, st_gen = CUDA.@fastmath log_posterior(half_quant.(z_i), x_i, Lux.testmode(st_i), t_k)
        ∇z = CUDA.@fastmath first(gradient(z_j -> sum(first(log_posterior(z_j, x_i, Lux.testmode(st_i), t_k))), half_quant.(z_i)))
        @reset st_i.ebm = st_ebm
        @reset st_i.gen = st_gen
        return full_quant.(logpos_z) ./ loss_scaling, full_quant.(∇z) ./ loss_scaling, st_i
    end 

    burn_in = 0
    η = st.η_init

    pos_before = CUDA.@fastmath first(log_posterior(half_quant.(z), x_t, Lux.testmode(st), temps)) ./ loss_scaling
    for i in 1:N
        z_cpu = cpu_device()(z)
        for k in 1:T
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
                domain, 
                temps
                )
        else
            z, η, η_prime, reversible, log_r, st = autoMALA_step(
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
                full_quant(Δη), 
                domain, 
                logpos_withgrad; 
                η_min=η_min, 
                η_max=η_max, 
                ε=full_quant(m.ε), 
                seq=seq)

            accept = (view(log_u,:,:,i) .< log_r) .* reversible
            z = z .* reshape(accept, 1, 1, S, T) + z .* reshape(1 .- accept, 1, 1, S, T)
            mean_η .= mean_η .+ η .* accept
            η = (η + η_prime) ./ 2
            num_acceptances .= num_acceptances .+ accept

            # Replica exchange Monte Carlo
            if i % RE_frequency == 0 && T > 1
                for t in 1:T-1

                    # Global swap criterion
                    z_hq = half_quant.(z)
                    ll_t, st_gen = log_llhood_fcn(view(z_hq,:,:,:,t), x, st.gen)
                    ll_t1, st_gen = log_llhood_fcn(view(z_hq,:,:,:,t+1), x, st_gen)
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
    pos_after = CUDA.@fastmath first(log_posterior(half_quant.(z), x_t, Lux.testmode(st), temps)) ./ loss_scaling
    m.verbose && println("t=$(temps) posterior change: $(dropdims(mean(pos_after - pos_before; dims=1); dims=1))")

    mean_η = clamp.(mean_η ./ num_acceptances, η_min, η_max)
    mean_η = ifelse.(isnan.(mean_η), st.η_init, mean_η) |> device
    @reset st.η_init = mean_η

    m.verbose && println("Acceptance rates: ", dropdims(mean(num_acceptances ./ (N - N_unadjusted); dims=1); dims=1))
    m.verbose && println("Mean step sizes: ", dropdims(mean(mean_η; dims=1); dims=1))
    
    return half_quant.(z), st, seed
end
end