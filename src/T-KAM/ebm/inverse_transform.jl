
module InverseTransformSampling

export sample_univariate, sample_mixture

using CUDA, LinearAlgebra, Random, Lux, LuxCUDA, ComponentArrays, ParallelStencil

using ..Utils

include("mixture_selection.jl")
using .MixtureChoice: choose_component

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

@parallel_indices (q, p, b) function interp_kernel!(
    z::AbstractArray{U,3},
    cdf::AbstractArray{U,3},
    grid::AbstractArray{U,2},
    rand_vals::AbstractArray{U,3},
    grid_size::Int,
)::Nothing where {U<:full_quant}
    rv = rand_vals[q, p, b]
    idx = 1

    # Manual searchsortedfirst over cdf[q, p, :] - potential thread divergence on GPU
    for j = 1:(grid_size+1)
        if cdf[q, p, j] >= rv
            idx = j
            break
        end
        idx = j
    end

    # Edge case 1: Random value is smaller than first CDF value
    if idx == 1
        z[q, p, b] = grid[p, 1]

        # Edge case 2: Random value is larger than last CDF value
    elseif idx > grid_size
        z[q, p, b] = grid[p, grid_size]

        # Interpolate into interval   
    else
        z1, z2 = grid[p, idx-1], grid[p, idx]
        cd1, cd2 = cdf[q, p, idx-1], cdf[q, p, idx]

        # Handle exact match without instability
        length = cd2 - cd1
        if length == 0
            z[q, p, b] = z1
        else
            z[q, p, b] = z1 + (z2 - z1) * ((rv - cd1) / length)
        end
    end
    return nothing
end

function sample_univariate(
    ebm::Lux.AbstractLuxLayer,
    num_samples::Int,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple;
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{AbstractArray{T,3},NamedTuple} where {T<:half_quant}

    cdf, grid, st_lyrnorm_new = ebm.quad(ebm, ps, st_kan, st_lyrnorm)
    grid_size = size(grid, 2)
    grid = full_quant.(grid)

    cdf = cat(
        pu(zeros(full_quant, ebm.q_size, ebm.p_size, 1)), # Add 0 to start of CDF
        cumsum(full_quant.(cdf); dims = 3), # Cumulative trapezium = CDF
        dims = 3,
    )

    rand_vals = pu(rand(rng, full_quant, 1, ebm.p_size, num_samples)) .* cdf[:, :, end]
    z = @zeros(ebm.q_size, ebm.p_size, num_samples)
    @parallel (1:ebm.q_size, 1:ebm.p_size, 1:num_samples) interp_kernel!(
        z,
        cdf,
        grid,
        rand_vals,
        grid_size,
    )
    return T.(z), st_lyrnorm_new
end

@parallel_indices (q, b) function interp_kernel_mixture!(
    z::AbstractArray{U,3},
    cdf::AbstractArray{U,3},
    grid::AbstractArray{U,2},
    rand_vals::AbstractArray{U,2},
    grid_size::Int,
)::Nothing where {U<:full_quant}
    rv = rand_vals[q, b]
    idx = 1

    # Manual searchsortedfirst over cdf[q, b, :] - potential thread divergence on GPU
    for j = 1:(grid_size+1)
        if cdf[q, b, j] >= rv
            idx = j
            break
        end
        idx = j
    end

    # Edge case 1: Random value is smaller than first CDF value
    if idx == 1
        z[q, 1, b] = grid[q, 1]

        # Edge case 2: Random value is larger than last CDF value
    elseif idx > grid_size
        z[q, 1, b] = grid[q, grid_size]

        # Interpolate into interval   
    else
        z1, z2 = grid[q, idx-1], grid[q, idx]
        cd1, cd2 = cdf[q, b, idx-1], cdf[q, b, idx]

        # Handle exact match without instability
        length = cd2 - cd1
        if length == 0
            z[q, 1, b] = z1
        else
            z[q, 1, b] = z1 + (z2 - z1) * ((rv - cd1) / length)
        end
    end
    return nothing
end

function sample_mixture(
    ebm::Lux.AbstractLuxLayer,
    num_samples::Int,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple;
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{AbstractArray{T,3},NamedTuple} where {T<:half_quant}
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
    """
    mask = choose_component(ps.dist.α, num_samples, ebm.q_size, ebm.p_size; rng = rng)

    cdf, grid, st_lyrnorm_new = ebm.quad(ebm, ps, st_kan, st_lyrnorm; component_mask = mask)
    grid_size = size(grid, 2)

    cdf = cat(
        pu(zeros(full_quant, ebm.q_size, num_samples, 1)), # Add 0 to start of CDF
        cumsum(full_quant.(cdf); dims = 3), # Cumulative trapezium = CDF
        dims = 3,
    )

    rand_vals = pu(rand(rng, full_quant, ebm.q_size, num_samples)) .* cdf[:, :, end]

    z = @zeros(ebm.q_size, 1, num_samples)
    @parallel (1:ebm.q_size, 1:num_samples) interp_kernel_mixture!(
        z,
        cdf,
        full_quant.(grid),
        rand_vals,
        grid_size,
    )
    return T.(z), st_lyrnorm_new
end

end
