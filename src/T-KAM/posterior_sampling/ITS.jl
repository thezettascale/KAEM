module LangevinSampling

export langevin_sampler, cross_entropy, l2

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
    Don't be fooled by the name, this is Inverse Transform Sampling.

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
    z, st_ebm, seed = m.prior.sample_z(m.prior, size(x)[end]*m.IS_samples, ps.ebm, st.ebm, seed)
    @reset st.ebm = st_ebm
    z = z .|> full_quant |> cpu_device()

    T, B = length(t), size(x)[end]
    output = reshape(z, m.prior.q_size, m.prior.p_size, B, m.IS_samples, 1)

    k = 1

    z = Array{full_quant}(undef, m.prior.q_size, m.prior.p_size, B, m.IS_samples, T)
    while k < T + 1
        prior_cdf, grid, st_ebm = m.prior.quad(m.prior, ps.ebm, st.ebm)
        g_ll = repeat(reshape(device(grid), 1, m.prior.p_size, size(grid, 2)), m.prior.q_size, 1, 1)
        lkhood_cdf, st_gen, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, g_ll; seed=seed, ε=m.ε)
        @reset st.ebm = st_ebm
        @reset st.gen = st_gen

        t_k, grid_size = t[k], size(grid, 2)
        lkhood_cdf = grid_size == size(prior_cdf, 3) ? exp.(t_k * lkhood_cdf) : exp.(t_k * (lkhood_cdf[:, 2:end] - lkhood_cdf[:, 1:end-1]))
        @tullio cdf[q, p, b, g] := prior_cdf[q, p, g] * lkhood_cdf[b, g]

        grid = grid .|> full_quant
        cdf = cat(
            zeros(full_quant, m.prior.q_size, m.prior.p_size, B, 1), # Add 0 to start of CDF
            cpu_device()(cumsum(cdf .|> full_quant; dims=4)), # Cumulative trapezium = CDF
            dims=4) 

        seed, rng = next_rng(seed)
        rand_vals = rand(rng, full_quant, 1, m.prior.p_size, B, m.IS_samples) .* cdf[:, :, :, end] 
        
        
        Threads.@threads for q in 1:m.prior.q_size
            for p in 1:m.prior.p_size
                for b in 1:B
                    for s in 1:m.IS_samples
                        rv = rand_vals[q, p, b, s]
                        idx = searchsortedfirst(cdf[q, p, b, :], rv)

                        idx = idx == 1 ? 2 : idx
                        idx = idx > grid_size ? grid_size : idx

                        z1, z2 = grid[p, idx-1], grid[p, idx]
                        cd1, cd2 = cdf[q, p, b, idx-1], cdf[q, p, b, idx]

                        z[q, p, b, s, k] = z1 + (z2 - z1) * ((rv - cd1) / (cd2 - cd1))
                    end
                end
            end
        end
        k += 1
    end
    return half_quant.(cat(output, z, dims=5)), st, seed
end
end