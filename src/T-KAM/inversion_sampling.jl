module InverseSampling

export sample_prior

using CUDA, KernelAbstractions, Tullio
using Random, Distributions, Lux, LuxCUDA, LinearAlgebra
using Flux: onehotbatch

include("../utils.jl")
include("univariate_functions.jl")
using .Utils: device, next_rng, removeZero, half_quant, full_quant
using .univariate_functions: fwd

function choose_component(
    α::AbstractArray{half_quant}, 
    num_samples::Int, 
    q_size::Int, 
    p_size::Int; 
    seed::Int=1
    )
    """
    Creates a one-hot mask for each mixture model, q, to select one component, p.
    
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
    mask = Array{half_quant}(undef, q_size, p_size, num_samples) 
    Threads.@threads for q in 1:q_size
        i = searchsortedfirst.(Ref(α[q, :]), rand_vals[q, :])

        # If necessary:
        replace!(i, p_size + 1 => p_size) # Upper bound

        mask[q, :, :] = onehotbatch(i, 1:p_size) .|> half_quant
    end

    return permutedims(mask, [3, 1, 2]) |> device, seed
end

function sample_prior(
    prior,
    num_samples::Int, 
    ps,
    st;
    seed::Int=1,
    ε::half_quant=eps(half_quant)
    )
    """
    Component-wise inverse transform sampling for the mixture ebm-prior.
    p = components of mixture model
    q = number of mixture models

    Args:
        prior: The mixture ebm-prior.
        ps: The parameters of the mixture ebm-prior.
        st: The states of the mixture ebm-prior.

    Returns:
        z: The samples from the mixture ebm-prior, (num_samples, q). 
        seed: The updated seed.
    """
    p_size = prior.fcns_qp[Symbol("$(prior.depth)")].out_dim
    q_size = prior.fcns_qp[Symbol("1")].in_dim
    
    # Categorical component selection (per sample, per outer sum dimension)
    component_mask, seed = choose_component(
        ps[Symbol("α")],
        num_samples,
        q_size,
        p_size;
        seed=seed
    )

    # Evaluate prior on grid [0,1]
    f_grid = prior.fcns_qp[Symbol("1")].grid'
    grid = f_grid |> cpu_device() .|> full_quant
    Δg = f_grid[2:end, :] - f_grid[1:end-1, :] .|> full_quant
    
    π_grid = prior.π_pdf(f_grid)
    grid_size = size(f_grid, 1)

    # Energy function of each component, q -> p
    for i in 1:prior.depth
        f_grid = fwd(prior.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], f_grid)
        f_grid = i == 1 ? reshape(f_grid, grid_size*q_size, size(f_grid, 3)) : dropdims(sum(f_grid, dims=2); dims=2)
    end
    f_grid = reshape(f_grid, grid_size, q_size, p_size)

    # Filter out components
    @tullio exp_fg[b, g, q] := (exp(f_grid[g, q, p]) * π_grid[g, q]) * component_mask[b, q, p]
    exp_fg = exp_fg .|> full_quant

    # CDF evaluated by trapezium rule for integration; 1/2 * (u(z_{i-1}) + u(z_i)) * Δx
    trapz = (permutedims(Δg[:,:,:], [3,1,2]) .* (exp_fg[:, 2:end, :] + exp_fg[:, 1:end-1, :])) ./ 2
    cdf = cumsum(trapz, dims=2) 
    cdf = cat(zeros(num_samples, 1, q_size), cpu_device()(cdf), dims=2) # Add 0 to start of CDF

    seed, rng = next_rng(seed)
    rand_vals = rand(rng, full_quant, num_samples, q_size) .* cdf[:, end, :] 
    
    z = Array{full_quant}(undef, num_samples, q_size)
    Threads.@threads for b in 1:num_samples
        for q in 1:q_size
            # First trapezium where CDF >= rand_val
            rv = rand_vals[b, q]
            idx = searchsortedfirst(cdf[b, :, q], rv) # Index of upper trapezium bound

            # If necessary:
            idx = idx == 1 ? 2 : idx
            idx = idx == grid_size + 1 ? grid_size : idx

            # Trapezium bounds
            z1, z2 = grid[idx-1, q], grid[idx, q] 
            cd1, cd2 = cdf[b, idx-1, q], cdf[b, idx, q] 

            # Linear interpolation
            z[b, q] = z1 + (z2 - z1) * ((rv - cd1) / removeZero(cd2 - cd1; ε=ε))
        end
    end

    return device(half_quant.(z)), seed
end

end