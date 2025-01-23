module InverseSampling

export sample_prior

using CUDA, KernelAbstractions, Tullio
using Random, Distributions, Lux, LuxCUDA, LinearAlgebra
using Flux: onehotbatch

include("../utils.jl")
include("univariate_functions.jl")
using .Utils: device, next_rng, removeZero, quant
using .univariate_functions: fwd

function choose_component(α::AbstractArray{quant}, num_samples::Int, q_size::Int, p_size::Int; seed::Int=1)
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
    rand_vals = rand(rng, Uniform(0,1), q_size, num_samples) 
    α = cumsum(softmax(α; dims=2); dims=2) |> cpu_device()

    # Find the index of the first cdf value greater than the random value
    mask = Array{quant}(undef, q_size, p_size, num_samples) 
    Threads.@threads for q in 1:q_size
        i = searchsortedfirst.(Ref(α[q, :]), rand_vals[q, :])
        replace!(i, p_size + 1 => p_size)
        mask[q, :, :] = onehotbatch(i, 1:p_size) .|> quant
    end

    return permutedims(mask, [3, 1, 2]) |> device, seed
end

function get_trap_bounds(idxs::AbstractArray{Int}, cdf::AbstractArray{quant})
    """Returns the CDF values bounding each trapezium defined by idxs."""
    Q, N = size(idxs)
    cdf = hcat(zeros(quant, N, 1, Q), cdf) # Zero prob

    cd1 = zeros(quant, N, Q) 
    cd2 = zeros(quant, N, Q) 
    for q in 1:Q
        for n in 1:N
            cd1[n, q] = cdf[n, idxs[q, n], q]
            cd2[n, q] = cdf[n, idxs[q, n] + 1, q]
        end
    end

    return device(cd1), device(cd2)
end

function interpolate_z(
    indices::AbstractArray{Int},
    cdf::AbstractArray{quant},
    rv::AbstractArray{quant},
    grid::AbstractArray{quant};
    seed::Int=1
    )
    """
    Returns samples of z from all mixture models, using linear interpolation
    to place grid point inside trapezium's interval.

    Args:
        indices: The indices of the trapeziums, (q,).
        cdf: The CDF values of the mixture ebm-prior, (grid_size-1, q).
        grid: The grid points of the mixture ebm-prior, (grid_size, q).
        Δg: The grid spacing of the mixture ebm-prior, (grid_size-1, q).
        num_samples: The number of samples to generate.
        seed: The seed for the random number generator.

    Returns:
        z: The samples from the mixture ebm-prior, (num_samples, q).
        seed: The updated seed.
    """
    # Get indices of trapeziums, and their corresponding cdfs
    cd1, cd2 = get_trap_bounds(indices, cdf)

    # Get trapezium bounds
    z1 = zeros(quant, size(cdf, 1), 0) |> device
    z2 = zeros(quant, size(cdf, 1), 0) |> device
    for (q, idx) in enumerate(eachrow(indices))
        z1 = hcat(z1, grid[idx, q:q])
        z2 = hcat(z2, grid[idx .+ 1, q:q])
    end

    # Linear interpolation
    return (z1 + (z2 - z1) .* ((rv - cd1) ./ removeZero(cd2 - cd1; ε=eps(eltype(cd1))))), seed
end

function sample_prior(
    prior,
    num_samples::Int, 
    ps,
    st;
    seed::Int=1,
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
    grid = prior.fcns_qp[Symbol("1")].grid'
    f_grid = grid
    Δg = f_grid[2:end, :] - f_grid[1:end-1, :] 
    π_grid = prior.π_pdf(f_grid)
    grid_size = size(f_grid, 1)
    for i in 1:prior.depth
        f_grid = fwd(prior.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], f_grid)
        f_grid = i == 1 ? reshape(f_grid, grid_size*q_size, size(f_grid, 3)) : dropdims(sum(f_grid, dims=2); dims=2)
    end
    f_grid = reshape(f_grid, grid_size, q_size, p_size)
    @tullio exp_fg[b, g, q] := exp(f_grid[g, q, p]) * π_grid[g, q] * component_mask[b, q, p]

    # CDF evaluated by trapezium rule for integration; 1/2 * (u(z_{i-1}) + u(z_i)) * Δx
    trapz = 5f-1 .* permutedims(Δg[:,:,:], [3,1,2]) .* (exp_fg[:, 2:end, :] + exp_fg[:, 1:end-1, :]) 
    cdf = cumsum(trapz, dims=2) 
    cdf = cdf ./ cdf[:, end:end, :] |> cpu_device() # Normalization
    cdf = clamp.(cdf, 0, 1) # Numerical issues with cdf values = 1.000001

     # Find index of trapezium where CDF > rand_val
    seed, rng = next_rng(seed)
    rand_vals = rand(rng, Uniform(0,1), num_samples, q_size) 
    idxs = Array{Int}(undef, q_size, num_samples)
    Threads.@threads for q in 1:q_size
        for b in 1:num_samples
            idxs[q, b] = searchsortedfirst(cdf[b, :, q], rand_vals[b, q])
        end
    end
    replace!(idxs, grid_size => grid_size - 1)

    z, seed = interpolate_z(idxs, cdf, device(rand_vals), grid; seed=seed)

    # Address numerical issues if any
    z = typeof(prior.π_0) == Uniform ? ifelse.(z .< 0, quant(0), z) |> device : z
    return z, seed
end

end