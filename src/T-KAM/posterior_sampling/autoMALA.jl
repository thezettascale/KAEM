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

abstract type Preconditioner end

struct IdentityPreconditioner <: Preconditioner end
struct DiagonalPreconditioner <: Preconditioner end

struct MixDiagonalPreconditioner{TR<:Real} <: Preconditioner
    p0::TR  # Proportion of zeros
    p1::TR  # Proportion of ones
    
    function MixDiagonalPreconditioner(p0::TR, p1::TR) where {TR<:Real}
        zero(TR) ≤ p0+p1 ≤ one(TR) || throw(ArgumentError("p0+p1 < 0 or p0+p1 > 1"))
        new{TR}(p0, p1)
    end
end

MixDiagonalPreconditioner() = MixDiagonalPreconditioner(1//3, 1//3)

# Default behavior - no preconditioning
function build_preconditioner!(
    dest::AbstractArray{T}, 
    ::IdentityPreconditioner,
    std_devs::AbstractArray{T}; 
    seed::Int=1
    ) where T
    fill!(dest, one(T))
    return dest
end

# Diagonal preconditioning
function build_preconditioner!(
    dest::AbstractArray{T}, 
    ::DiagonalPreconditioner,
    std_devs::AbstractArray{T}; 
    seed::Int=1
    ) where T
    @. dest = ifelse(iszero(std_devs), one(T), one(T) / std_devs)
    return dest
end

# Mixed diagonal preconditioning
function build_preconditioner!(
    dest::AbstractArray{T}, 
    prec::MixDiagonalPreconditioner,
    std_devs::AbstractArray{T}; 
    seed::Int=1
    ) where T
    seed, rng = next_rng(seed)
    u = rand(rng, T)
    
    if u ≤ prec.p0
        # Use inverse standard deviations
        @. dest = ifelse(iszero(std_devs), one(T), one(T) / std_devs)
    elseif u ≤ prec.p0 + prec.p1
        # Use identity
        fill!(dest, one(T))
    else
        # Random mixture
        seed, rng = next_rng(seed)
        mix = rand(rng, T)
        rmix = one(T) - mix
        @. dest = ifelse(iszero(std_devs), 
                        one(T), 
                        mix + rmix / std_devs)
    end
    return dest
end

# This is transformed momentum!
function sample_momentum(
    z::AbstractArray{full_quant};
    seed::Int=1,
    preconditioner::Preconditioner=MixDiagonalPreconditioner(),
    ε::full_quant=eps(full_quant)
    )
    Q, P, S = size(z)
    z_cpu = cpu_device()(z)
    
    # Compute M^{1/2}
    z_reshaped = reshape(z_cpu, Q*P, S)
    μ = mean(z_reshaped, dims=2)
    Σ = sqrt.(@views sum((z_reshaped .- μ).^2, dims=2) ./ (S-1))
    
    # Initialize mass matrix (M^{1/2})
    M = ones(full_quant, Q*P)
    build_preconditioner!(M, preconditioner, vec(Σ); seed=seed)
    M = reshape(M, Q, P)
    
    # Sample y ~ N(0,I) directly (transformed momentum)
    seed, rng = next_rng(seed)
    y = randn(rng, full_quant, Q, P, S)
    
    return device(y), device(M), seed
end

function safe_step_size_update(
    η::AbstractVector{full_quant}, 
    δ::AbstractVector{Int}, 
    Δη::full_quant
    )
    η_new = η .* Δη.^δ
    return ifelse.(isfinite.(η_new), η_new, η)
end

function check_reversibility(
    ẑ::AbstractArray{full_quant}, 
    z::AbstractArray{full_quant}, 
    η::AbstractVector{full_quant}, 
    η_prime::AbstractVector{full_quant};
    tol::full_quant=full_quant(1e-6)
    )
    # pos_diff = dropdims(maximum(abs.(ẑ - z); dims=(1,2)); dims=(1,2)) .< tol * maximum(abs.(z))
    step_diff = abs.(η - η_prime) .< tol .* η
    return step_diff
end

function leapfrop_proposal(
    z::AbstractArray{full_quant},
    x::AbstractArray{half_quant},
    st,
    logpos_z::AbstractVector{full_quant},
    ∇z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},  # This is y = M^{-1/2}p
    M::AbstractArray{full_quant},         # This is M^{1/2}
    η::AbstractVector{full_quant},
    logpos_withgrad::Function
    )
    """
    Implements preconditioned Hamiltonian dynamics with transformed momentum:
    y*(x,y)   = y  + (eps/2)M^{-1/2}grad(log pi)(x)
    x'(x,y*)  = x  + eps M^{-1/2}y*
    y'(x',y*) = y* + (eps/2)M^{-1/2}grad(log pi)(x')
    """
    # Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad)
    @tullio p_in[q,p,s] := momentum[q,p,s] + (η[s]/2) * ∇z[q,p,s] / M[q,p]

    # Full step position update (x' = x + eps M^{-1/2}y*)
    @tullio ẑ[q,p,s] := z[q,p,s] + η[s] * p_in[q,p,s] / M[q,p]

    # Get gradient at new position
    logpos_ẑ, ∇ẑ, st = logpos_withgrad(ẑ, x, st)

    # Last half-step momentum update
    @tullio p_out[q,p,s] := p_in[q,p,s] + (η[s]/2) * ∇ẑ[q,p,s] / M[q,p]

    # Compute Hamiltonian difference for transformed momentum
    # H(x,y) = -log(pi(x)) + (1/2)||p||^2 since p ~ N(0,I)
    log_r = logpos_ẑ - logpos_z - (dropdims(sum(p_out.^2; dims=(1,2)) - sum(momentum.^2; dims=(1,2)); dims=(1,2)) ./ 2)

    return ẑ, logpos_ẑ, ∇ẑ, -p_out, log_r, st
end

function select_step_size(
    log_a::AbstractArray{full_quant},
    log_b::AbstractArray{full_quant},
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
    η_min::full_quant=full_quant(1e-5),
    η_max::full_quant=full_quant(1),
    seq::Bool=false
    )
    
    ẑ, logpos_ẑ, ∇ẑ, p̂, log_r, st = leapfrop_proposal(z, x, st, logpos_z, ∇z, momentum, M, η_init, logpos_withgrad)

    δ = (log_r .>= log_b) - (log_r .<= log_a)
    active_chains = findall(δ .!= 0)
    isempty(active_chains) && return ẑ, logpos_ẑ, ∇ẑ, p̂, η_init, log_r, st
    
    geq_bool = log_r .>= log_b

    while !isempty(active_chains)
        η_init[active_chains] = safe_step_size_update(
            η_init[active_chains], 
            δ[active_chains], 
            Δη
        )
        
        x_active = seq ? view(x, :, :, active_chains) : view(x, :, :, :, active_chains)
        ẑ_active, logpos_ẑ_active, ∇ẑ_active, p̂_active, log_r_active, st = 
            leapfrop_proposal(
                view(z,:,:,active_chains), 
                x_active,
                st, 
                view(logpos_z,active_chains), 
                view(∇z,:,:,active_chains), 
                view(momentum,:,:,active_chains),
                M,
                view(η_init,active_chains), 
                logpos_withgrad
            )
        
        # Update active chain results
        ẑ[:,:,active_chains] = ẑ_active
        logpos_ẑ[active_chains] = logpos_ẑ_active  
        ∇ẑ[:,:,active_chains] = ∇ẑ_active
        p̂[:,:,active_chains] = p̂_active
        log_r[active_chains] = log_r_active

        # Update which chains still need adjustment with improved stability checks
        δ[active_chains] = ifelse.(δ[active_chains] .== 1 .&& log_r[active_chains] .< log_b[active_chains], 0, δ[active_chains])
        δ[active_chains] = ifelse.(δ[active_chains] .== -1 .&& log_r[active_chains] .> log_a[active_chains], 0, δ[active_chains])
        δ[active_chains] = ifelse.(η_min .< η_init[active_chains] .< η_max, δ[active_chains], 0)
        active_chains = findall(δ .!= 0)
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
    st,
    logpos_z::AbstractVector{full_quant},
    ∇z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η_init::AbstractVector{full_quant},
    Δη::full_quant,
    logpos_withgrad::Function;
    η_min::full_quant=full_quant(1e-5),
    η_max::full_quant=full_quant(1),
    ε::full_quant=eps(full_quant),
    seq::Bool=false
    )
    
    ẑ, logpos_ẑ, ∇ẑ, p̂, η, log_r, _ = select_step_size(
        log_a, log_b, z, x, st, logpos_z, ∇z, momentum, M, η_init, Δη, 
        logpos_withgrad; η_min=η_min, η_max=η_max, seq=seq
    )
    
    z_rev, _, _, _, η_prime, _, st = select_step_size(
        log_a, log_b, ẑ, x, st, logpos_ẑ, ∇ẑ, p̂, M, η_init, Δη, 
        logpos_withgrad; η_min=η_min, η_max=η_max, seq=seq
    )
    
    reversible = check_reversibility(z, z_rev, η, η_prime; tol=ε)
    return ẑ, η, η_prime, reversible, log_r, st
end

function transform_to_unbounded(z::AbstractArray{T}, domain::Tuple{T,T}, ε::T) where T
    a, b = domain
    # Transform [a,b] -> [-∞,∞] using logit
    z_unbounded = log.((z .- a) ./ (b .- z) .+ ε)
    # Compute log|det(J)| for the transform
    log_det_J = dropdims(sum(log.((b .- a) ./ ((z .- a) .* (b .- z)) .+ ε); dims=(1,2)); dims=(1,2))
    return z_unbounded, log_det_J
end

function transform_to_bounded(z_unbounded::AbstractArray{T}, domain::Tuple{T,T}, ε::T) where T
    a, b = domain
    # Transform [-∞,∞] -> [a,b] using sigmoid
    z = a .+ (b .- a) .* sigmoid_fast.(z_unbounded)
    # Compute log|det(J)| for the inverse transform
    log_det_J = -dropdims(sum(log.((b .- a) ./ ((z .- a) .* (b .- z)) .+ ε); dims=(1,2)); dims=(1,2))
    return z, log_det_J
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
    domain = m.prior.fcns_qp[Symbol("1")].grid_range
    
    # Initialize from prior and transform to unbounded space
    z, st_ebm, seed = m.prior.sample_z(m.prior, size(x)[end], ps.ebm, st.ebm, seed)
    z_unbounded, log_det_J = transform_to_unbounded(full_quant.(z), full_quant.(domain), full_quant(m.ε))
    loss_scaling = m.loss_scaling |> full_quant

    T, Q, P, S = length(t), size(z)...
    output = reshape(z, Q, P, S, 1)

    # Avoid looped stochasticity
    seed, rng = next_rng(seed)
    log_u = log.(rand(rng, full_quant, T, N, S)) |> device
    seed, rng = next_rng(seed)
    ratio_bounds = log.(rand(rng, Uniform(0,1), T, N, S, 2)) .|> full_quant |> device

    seq = m.lkhood.seq_length > 1
    ll_fn = seq ? (x_i, y_i) -> dropdims(sum(cross_entropy(x_i, y_i; ε=m.ε); dims=1); dims=1) : (x_i, y_i) -> dropdims(sum(l2(x_i, y_i; ε=m.ε); dims=(1,2,3)); dims=(1,2,3))

    function log_posterior(z_i::AbstractArray{half_quant}, x_i::AbstractArray{half_quant}, st_i, t_k::half_quant)

        # Bound to prior domain
        z_bounded, log_det_J = transform_to_bounded(z_i, domain, m.ε)

        lp, st_ebm = log_prior(m.prior, z_bounded, ps.ebm, st_i.ebm; ε=m.ε)
        x̂, st_gen = m.lkhood.generate_from_z(m.lkhood, ps.gen, st_i.gen, z_bounded)
        x̂ = m.lkhood.output_activation(x̂) 
        logpos = lp + t_k * ll_fn(x̂, x_i) / (2*m.lkhood.σ_llhood^2) + log_det_J # Jacobian correction
        return logpos .* m.loss_scaling, st_ebm, st_gen
    end

    k = 1
    num_acceptances = zeros(Int, T, S) |> device
    mean_η = zeros(full_quant, T, S) |> device

    while k < T + 1
        
        logpos_withgrad = (z_i, x_i, st_i) -> begin
            logpos_z, st_ebm, st_gen = CUDA.@fastmath log_posterior(half_quant.(z_i), x_i, Lux.testmode(st_i), t[k])
            ∇z = CUDA.@fastmath first(gradient(z_j -> sum(first(log_posterior(z_j, x_i, Lux.testmode(st_i), t[k]))), half_quant.(z_i)))
            @reset st_i.ebm = st_ebm
            @reset st_i.gen = st_gen
            return full_quant.(logpos_z) ./ loss_scaling, full_quant.(∇z) ./ loss_scaling, st_i
        end

        burn_in = 0
        η = st.η_init[k, :]
        m.verbose && println("t=$(t[k]) posterior before update: ", sum(first(log_posterior(half_quant.(z), x, Lux.testmode(st), t[k]))) ./ loss_scaling)

        for i in 1:N
            momentum, M, seed = sample_momentum(z_unbounded; seed=seed, ε=full_quant(m.ε))
            log_a, log_b = dropdims(minimum(ratio_bounds[k, i, :, :]; dims=2); dims=2), dropdims(maximum(ratio_bounds[k, i, :, :]; dims=2); dims=2)
            logpos_z, ∇z, st = logpos_withgrad(z_unbounded, x, st)

            if burn_in < N_unadjusted
                z_unbounded, logpos_ẑ, ∇ẑ, p̂, log_r, st = leapfrop_proposal(z_unbounded, x, st, logpos_z, ∇z, momentum, M, η, logpos_withgrad)
                burn_in += 1
            else
                ẑ_unbounded, η, η_prime, reversible, log_r, st = autoMALA_step(
                    log_a, 
                    log_b, 
                    z_unbounded, 
                    x, st, 
                    logpos_z, 
                    ∇z, 
                    momentum, 
                    M, 
                    η, 
                    Δη, 
                    logpos_withgrad; 
                    η_min=η_min, 
                    η_max=η_max, 
                    ε=full_quant(m.ε), 
                    seq=seq
                    )

                accept = (view(log_u,k,i,:) .< log_r) .* reversible
                z_unbounded = ẑ_unbounded .* reshape(accept, 1, 1, :) + z_unbounded .* reshape(1 .- accept, 1, 1, :)
                mean_η[k, :] .= mean_η[k, :] .+ η .* accept
                η = (η + η_prime) ./ 2
                num_acceptances[k, :] .= num_acceptances[k, :] .+ accept
            end
        end
        m.verbose && println("t=$(t[k]) posterior after update: ", sum(first(log_posterior(half_quant.(z), x, Lux.testmode(st), t[k]))) ./ loss_scaling)
        
        z_out, _ = transform_to_bounded(z_unbounded, full_quant.(domain), full_quant(m.ε))
        output = cat(output, reshape(z_out, Q, P, S, 1); dims=4)
        k += 1
    end

    mean_η = clamp.(mean_η ./ num_acceptances, η_min, η_max)
    mean_η = ifelse.(isnan.(mean_η), st.η_init, mean_η) |> device
    @reset st.η_init = mean_η

    m.verbose && println("Acceptance rates: ", dropdims(mean(num_acceptances ./ (N - N_unadjusted); dims=2); dims=2))
    m.verbose && println("Mean step sizes: ", mean_η)

    return half_quant.(output), st, seed
end
end