
module InverseTransformSampling

export sample_univariate, sample_mixture

using CUDA, KernelAbstractions, Tullio, LinearAlgebra, Random, Lux, LuxCUDA
using NNlib: softmax

include("../../utils.jl")
include("../log_prior_fcns.jl")
using .Utils: device, next_rng, half_quant, full_quant, fq
using .LogPriorFCNs: prior_fwd
using Flux: onehotbatch

function choose_component(
    α::AbstractArray{T}, 
    num_samples::Int, 
    q_size::Int, 
    p_size::Int; 
    seed::Int=1
    ) where {T<:half_quant}
    """
    Creates a one-hot mask for mixture model, q, to select one component, p.

    Args:
        alpha: The mixture proportions, (q, p).
        num_samples: The number of samples to generate.
        q_size: The number of mixture models.
        seed: The seed for the random number generator.

    Returns:
        chosen_components: The one-hot mask for each mixture model, (num_samples, q, p).
        seed: The updated seed.
    """
    seed, rng = next_rng(seed)
    rand_vals = rand(rng, full_quant, q_size, num_samples) 
    α = cumsum(softmax(α .|> full_quant; dims=2); dims=2) |> cpu_device() 

    # Find the index of the first cdf value greater than the random value
    mask = Array{T}(undef, q_size, p_size, num_samples) 
    Threads.@threads for q in 1:q_size
        i = searchsortedfirst.(Ref(α[q, :]), rand_vals[q, :])
        replace!(i, p_size + 1 => p_size) # Edge case 
        mask[q, :, :] = onehotbatch(i, 1:p_size) .|> T
    end
    
    return mask |> device, seed
end

function sample_univariate(
    ebm,
    num_samples::Int, 
    ps,
    st;
    seed::Int=1,
    ε::T=eps(T)
    ) where {T<:half_quant}

    cdf, grid, st = ebm.quad(ebm, ps, st, nothing)
    grid_size = size(grid, 2)
    grid = grid .|> full_quant

    cdf = cat(
        zeros(full_quant, ebm.q_size, ebm.p_size, 1), # Add 0 to start of CDF
        cpu_device()(cumsum(cdf .|> full_quant; dims=3)), # Cumulative trapezium = CDF
        dims=3) 

    seed, rng = next_rng(seed)
    rand_vals = rand(rng, full_quant, 1, ebm.p_size, num_samples) .* cdf[:, :, end] 
    
    z = Array{full_quant}(undef, ebm.q_size, ebm.p_size, num_samples)
    Threads.@threads for q in 1:ebm.q_size
        for p in 1:ebm.p_size
            for b in 1:num_samples
                # First trapezium where CDF >= rand_val
                rv = rand_vals[q, p, b]
                idx = searchsortedfirst(cdf[q, p, :], rv) # Index of upper trapezium bound

                # Edge cases
                idx = idx == 1 ? 2 : idx
                idx = idx > grid_size ? grid_size : idx

                # Trapezium bounds
                z1, z2 = grid[p, idx-1], grid[p, idx] 
                cd1, cd2 = cdf[q, p, idx-1], cdf[q, p, idx]
 
                # Linear interpolation
                z[q, p, b] = z1 + (z2 - z1) * ((rv - cd1) / (cd2 - cd1))
            end
        end
    end

    return device(T.(z)), st, seed
end

function sample_mixture(
    ebm,
    num_samples::Int, 
    ps,
    st;
    seed::Int=1,
    ε::T=eps(T)
    ) where {T<:half_quant}
    """
    Component-wise inverse transform sampling for the ebm-prior.
    p = components of model
    q = number of models

    Args:
        prior: The ebm-prior.
        ps: The parameters of the ebm-prior.
        st: The states of the ebm-prior.

    Returns:
        z: The samples from the ebm-prior, (num_samples, q). 
        seed: The updated seed.
    """
    mask, seed = choose_component(
        ps[Symbol("α")],
        num_samples,
        ebm.q_size,
        ebm.p_size;
        seed=seed
    )

    cdf, grid, st = ebm.quad(ebm, ps, st, mask)
    grid_size = size(grid, 2)
    grid = grid .|> full_quant

    cdf = cat(
        zeros(full_quant, ebm.q_size, num_samples, 1), # Add 0 to start of CDF
        cpu_device()(cumsum(cdf .|> full_quant; dims=3)), # Cumulative trapezium = CDF
        dims=3) 

    seed, rng = next_rng(seed)
    rand_vals = rand(rng, full_quant, ebm.q_size, num_samples) .* cdf[:, :, end] 

    z = Array{full_quant}(undef, ebm.q_size, 1, num_samples)
    Threads.@threads for q in 1:ebm.q_size
        for b in 1:num_samples
            # First trapezium where CDF >= rand_val
            rv = rand_vals[q, b]
            idx = searchsortedfirst(cdf[q, b, :], rv) # Index of upper trapezium bound

            # Edge cases
            idx = idx == 1 ? 2 : idx
            idx = idx > grid_size ? grid_size : idx

            # Trapezium bounds
            z1, z2 = grid[q, idx-1], grid[q, idx] 
            cd1, cd2 = cdf[q, b, idx-1], cdf[q, b, idx]

            # Linear interpolation
            z[q, 1, b] = z1 + (z2 - z1) * ((rv - cd1) / (cd2 - cd1))
        end
    end

    return device(T.(z)), st, seed
end

end