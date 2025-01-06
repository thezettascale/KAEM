module InverseSampling

export sample_prior

using CUDA, KernelAbstractions, Tullio
using Random, Distributions, Lux, LuxCUDA, LinearAlgebra
using Flux: onehotbatch

include("../utils.jl")
include("univariate_functions.jl")
using .Utils: device, next_rng, removeZero
using .univariate_functions: fwd

function choose_component(alpha, num_samples, q_size, p_size; seed=1)
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
    
    function categorical_mask(α, rv)
        """Returns sampled indices from a categorical distribution on alpha."""
        idxs = map(u -> findfirst(x -> x >= u, α), rv)
        idxs = reduce(vcat, idxs)
        idxs = ifelse.(isnothing.(idxs), p_size, idxs)
        idxs = collect(Float32, onehotbatch(idxs, 1:p_size))   
        return permutedims(idxs[:,:,:], [2, 3, 1])
    end
    
    chosen_components = map(i -> categorical_mask(view(alpha, i, :), view(rand_vals, i, :)), 1:q_size)
    return reduce(hcat, chosen_components) |> device, seed
end

function get_z_with_noise(
    indices::AbstractArray,
    grid::AbstractArray,
    Δg::AbstractArray;
    seed::Int=1
)
    """
    Returns samples of z from all mixture models, using noise 
    to place grid point inside trapezium's interval.
    
    Args:
        indices: The indices of the trapeziums, (q,).
        grid: The grid points of the mixture ebm-prior, (grid_size, q).
        Δg: The grid spacing of the mixture ebm-prior, (grid_size-1, q).
        num_samples: The number of samples to generate.
        seed: The seed for the random number generator.

    Returns:
        z: The samples from the mixture ebm-prior, (num_samples, q).
        seed: The updated seed.
    """
    function get_noise(ub::Float32)
        """Returns random noise in the interval [0, ub], to place 
        grid point inside trapezium's interval."""
        seed, rng = next_rng(seed)
        return rand(rng, Uniform(0, ub))
    end

    z = zeros(Float32, length(first(indices)), 0) |> device
    for (q, idx) in enumerate(indices)
        noise = get_noise.(Δg[idx, q:q]) |> device
        z = hcat(z, grid[idx, q:q] .+ noise)
    end
    return z, seed 
end

function get_trap_bounds(idxs, cdf)
    """Returns the CDF values bounding each trapezium defined by idxs."""
    zero_prob = zeros(Float32, size(cdf, 1), 1, size(cdf, 3)) |> device
    cdf = hcat(zero_prob, cdf)

    cd1, cd2 = zeros(Float32, size(cdf,1 ), 0) |> device, zeros(Float32, size(cdf, 1), 0) |> device
    for (q, idx) in enumerate(idxs)
        bound1 = reduce(vcat,[cdf[i:i, g, q:q] for (i, g) in enumerate(idx)])
        bound2 = reduce(vcat,[cdf[i:i, g, q:q] for (i, g) in enumerate(idx .+ 1)])
        cd1, cd2 = hcat(cd1, bound1), hcat(cd2, bound2)
    end
    return cd1, cd2
end

function get_z_with_interpolation(
    indices::AbstractArray,
    cdf::AbstractArray,
    rv::AbstractArray,
    grid::AbstractArray;
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
    z1 = reduce(hcat, [grid[indices[i], i:i] for i in eachindex(indices)])
    z2 = reduce(hcat, [grid[indices[i] .+ 1, i:i] for i in eachindex(indices)])

    # Linear interpolation
    return (z1 + (z2 - z1) .* ((rv - cd1) ./ removeZero(cd2 - cd1; ε=eps(eltype(cd1))))), seed
end

function sample_prior(
    prior,
    num_samples::Int, 
    ps,
    st;
    seed::Int=1,
    interpolation::Bool=false
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
        cpu_device()(cumsum(softmax(ps[Symbol("α")]; dims=2); dims=2)),
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
    cdf = cumsum(trapz, dims=2) ./ sum(trapz, dims=2)

    # Inverse transform sampling
    seed, rng = next_rng(seed)
    rand_vals = rand(rng, Uniform(0,1), num_samples, q_size) |> device
    @tullio geq_indices[b, g, q] := cdf[b, g, q] >= rand_vals[b, q]

    function grid_index(q)
    """Returns index of trapz where CDF > rand_val from a given mixture model, q."""
        idxs = map(i -> findfirst(view(geq_indices, i, :, q)), 1:num_samples)
        idxs = reduce(vcat, idxs)
        idxs = ifelse.(isnothing.(idxs), grid_size-1, idxs)
        return idxs
    end
     
    indices = map(grid_index, 1:q_size)
    interpolation && return get_z_with_interpolation(indices, cdf, rand_vals, grid; seed=seed)
    return get_z_with_noise(indices, grid, cpu_device()(Δg); seed=seed)
end

end