module LangevinSampling

export langevin_sampler

using CUDA, KernelAbstractions, Tullio, LinearAlgebra, Random, Lux, LuxCUDA, Distributions, Accessors, Statistics
using Zygote: gradient

include("../utils.jl")
include("EBM_prior.jl")
using .Utils: device, next_rng, half_quant, full_quant, fq
using .ebm_ebm_prior: log_prior

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
    z, st_ebm, seed = m.prior.sample_z(m.prior, size(x)[end], ps.ebm, st.ebm, seed)
    @reset st.ebm = st_ebm
    z = z .|> full_quant
    loss_scaling = m.loss_scaling |> full_quant

    η = mean(st.η_init)

    T, Q, P, S = length(t), size(z)...
    output = reshape(z, Q, P, S, 1)

    # Avoid looped stochasticity
    seed, rng = next_rng(seed)
    noise = randn(rng, full_quant, Q, P, S, N, T)

    ll_fn = m.lkhood.seq_length > 1 ? (x,y) -> cross_entropy(x, y; ε=m.ε) : (x,y) -> l2(x, y; ε=m.ε)

    function log_posterior(z_i::AbstractArray{half_quant}, st_i, t_k::half_quant)
        lp, st_ebm = log_prior(m.prior, z_i, ps.ebm, st_i.ebm; ε=m.ε)
        x̂, st_gen = m.lkhood.generate_from_z(m.lkhood, ps.gen, st_i.gen, z_i)
        x̂ = m.lkhood.output_activation(x̂) 
        logpos = sum(lp) + sum(t_k .* ll_fn(x, x̂) ./ (2*m.lkhood.σ_llhood^2))
        return logpos .* m.loss_scaling, st_ebm, st_gen
    end

    k = 1

    while k < T + 1
        logpos_grad = (z_i) -> begin
            logpos_z, st_ebm, st_gen = CUDA.@fastmath log_posterior(half_quant.(z_i), Lux.testmode(st), t[k])
            ∇z = CUDA.@fastmath first(gradient(z_j -> sum(first(log_posterior(z_j, Lux.testmode(st), t[k]))), half_quant.(z_i)))
            @reset st.ebm = st_ebm
            @reset st.gen = st_gen
            return full_quant.(∇z) ./ loss_scaling
        end

        m.verbose && println("t=$(t[k]) posterior before update: ", full_quant(first(log_posterior(half_quant.(z), Lux.testmode(st), t[k]))) ./ loss_scaling)
        for i in 1:N
            ξ = device(noise[:,:,:,i,k])            
            z = z + η .* logpos_grad(z) .+ sqrt(2 * η) .* ξ
        end
        m.verbose && println("t=$(t[k]) posterior after update: ", full_quant(first(log_posterior(half_quant.(z), Lux.testmode(st), t[k]))) ./ loss_scaling)
        output = cat(output, reshape(z, Q, P, S, 1); dims=4)
        k += 1
    end

    return half_quant.(output), st, seed
end

end