module LangevinSampling

export langevin_sampler

using CUDA, KernelAbstractions, Tullio, LinearAlgebra, Random, Lux, LuxCUDA, Distributions, Accessors, Statistics
using Zygote: gradient

include("../../utils.jl")
include("../EBM_prior.jl")
include("../KAN_likelihood.jl")
using .Utils: device, next_rng, half_quant, full_quant, fq
using .ebm_ebm_prior: log_prior
using .KAN_likelihood: log_likelihood
function cross_entropy(x::AbstractArray{half_quant}, y::AbstractArray{half_quant}; ε::half_quant=eps(half_quant))
    return log.(x .+ ε) .* y ./ size(x, 1)
end

function l2(x::AbstractArray{half_quant}, y::AbstractArray{half_quant}; ε::half_quant=eps(half_quant))
    return -(x - y).^2
end

function langevin_sampler(
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
    seed::Int=1,
    max_zero_accept_iters::Int=50,
    RE_frequency::Int=10,
    ) where {T<:half_quant, U<:full_quant}
    """
    Unadjusted Langevin Algorithm (ULA) sampler to generate posterior samples.

    Args:
        m: The model.
        ps: The parameters of the model.
        st: The states of the model.
        x: The data.
        t: The temperatures if using Thermodynamic Integration.
        N: The number of iterations.
        seed: The seed for the random number generator.

        
    Unused arguments:
        N_unadjusted: The number of unadjusted iterations.
        Δη: The step size increment.
        η_min: The minimum step size.
        η_max: The maximum step size.

    Returns:
        The posterior samples.
    """
    # Initialize from prior
    z, st_ebm, seed = m.prior.sample_z(m.prior, size(x)[end]*length(temps), ps.ebm, st.ebm, seed)
    @reset st.ebm = st_ebm
    z = z .|> U
    loss_scaling = m.loss_scaling |> U

    # Get domain bounds
    domain = U.(m.prior.fcns_qp[Symbol("1")].grid_range)

    η = mean(st.η_init)

    T_length, Q, P, S = length(temps), size(z)[1:2]..., size(x)[end]
    z = reshape(z, Q, P, S, T_length)

    # Avoid looped stochasticity
    seed, rng = next_rng(seed)
    noise = randn(rng, U, Q, P, S, T_length, N)
    seed, rng = next_rng(seed)
    log_u_swap = log.(rand(rng, U, S, T_length, N)) |> device

    seq = m.lkhood.seq_length > 1
    ll_fn = seq ? (y_i) -> dropdims(sum(cross_entropy(y_i, x; ε=m.ε); dims=1); dims=1) : (y_i) -> dropdims(sum(l2(y_i, x; ε=m.ε); dims=(1,2,3)); dims=(1,2,3))
    
    # log_llhood_fcn = (z_i, st_gen) -> begin
    #     ll, st_gen, seed = log_likelihood(m.lkhood, ps.gen, st_gen, x, z_i; seed=seed, ε=m.ε)
    #     return ll, st_gen
    # end

    log_llhood_fcn = (z_i, st_gen) -> begin
        x̂, st_gen = m.lkhood.generate_from_z(m.lkhood, ps.gen, st_gen, z_i)
        return ll_fn(m.lkhood.output_activation(x̂)) ./ (2*m.lkhood.σ_llhood^2), st_gen
    end


    function log_posterior(z_i::AbstractArray{T}, st_i)
        logpos_tot = zero(T)
        st_ebm, st_gen = st_i.ebm, st_i.gen
        for k in 1:T_length
            z_k = view(z_i, :, :, :, k)
            lp, st_ebm = log_prior(m.prior, z_k, ps.ebm, st_ebm; ε=m.ε)
            ll, st_gen = log_llhood_fcn(z_k, st_gen)
            logpos_tot += sum(lp) + sum(view(temps, k) .* ll)
        end
        return logpos_tot * m.loss_scaling, st_ebm, st_gen
    end

    logpos_grad = (z_i) -> begin
        logpos_z, st_ebm, st_gen = CUDA.@fastmath log_posterior(T.(z_i), Lux.testmode(st))
        ∇z = CUDA.@fastmath first(gradient(z_j -> sum(first(log_posterior(z_j, Lux.testmode(st)))), T.(z_i)))
        @reset st.ebm = st_ebm
        @reset st.gen = st_gen
        return U.(∇z) ./ loss_scaling
    end

    pos_before = CUDA.@fastmath first(log_posterior(T.(z), Lux.testmode(st))) ./ loss_scaling
    for i in 1:N
        ξ = device(noise[:,:,:,:,i])
        z = z + η .* logpos_grad(z) .+ sqrt(2 * η) .* ξ

        # Reflect at boundaries
        reflect_low = z .< first(domain)
        reflect_high = z .> last(domain)
        z = ifelse.(reflect_low, 2*first(domain) .- z, z)
        z = ifelse.(reflect_high, 2*last(domain) .- z, z)

        if i % RE_frequency == 0 && T_length > 1
            z_hq = T.(z)
            for t in 1:T_length-1
                ll_t, st_gen = log_llhood_fcn(view(z_hq,:,:,:,t), st.gen)
                ll_t1, st_gen = log_llhood_fcn(view(z_hq,:,:,:,t+1), st_gen)
                log_swap_ratio = dropdims(sum((view(temps,t+1) - view(temps,t)) .* (ll_t - ll_t1); dims=1); dims=1)
                swap = view(log_u_swap,:,t,i) .< log_swap_ratio
                @reset st.gen = st_gen

                # Swap samples where accepted
                z[:,:,:,t] .= z[:,:,:,t] .* reshape(swap, 1, 1, S) + z[:,:,:,t+1] .* reshape(1 .- swap, 1, 1, S)
                z[:,:,:,t+1] .= z[:,:,:,t+1] .* reshape(swap, 1, 1, S) + z[:,:,:,t] .* reshape(1 .- swap, 1, 1, S)
            end
        end
    end

    pos_after = CUDA.@fastmath first(log_posterior(T.(z), Lux.testmode(st))) ./ loss_scaling
    m.verbose && println("Posterior change: $(pos_after - pos_before)")

    return T.(z), st, seed
end

end