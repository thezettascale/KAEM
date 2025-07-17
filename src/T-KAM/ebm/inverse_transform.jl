
module InverseTransformSampling

export sample_univariate, sample_mixture, gausslegendre_quadrature, trapezium_quadrature

using NNlib: softmax
using CUDA,
    KernelAbstractions,
    LinearAlgebra,
    Random,
    Lux,
    LuxCUDA,
    Tullio,
    ComponentArrays,
    ParallelStencil

include("../../utils.jl")
include("log_prior_fcns.jl")
using .Utils: device, next_rng, half_quant, full_quant, fq
using .LogPriorFCNs: prior_fwd
using Flux: onehotbatch

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

function trapezium_quadrature(
    ebm::Any,
    ps::ComponentArray{T},
    st::NamedTuple;
    ε::T = eps(half_quant),
    component_mask::Union{AbstractArray{<:half_quant},Nothing} = nothing,
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple} where {T<:half_quant}
    """Trapezoidal rule for numerical integration"""

    # Evaluate prior on grid [0,1]
    f_grid = st[Symbol("1")].grid
    grid = f_grid
    Δg = f_grid[:, 2:end] - f_grid[:, 1:(end-1)]

    π_grid =
        ebm.prior_type == "learnable_gaussian" ? ebm.π_pdf(f_grid', ps, ε) :
        ebm.π_pdf(f_grid, ε)
    π_grid = ebm.prior_type == "learnable_gaussian" ? π_grid' : π_grid

    # Energy function of each component
    f_grid, st = prior_fwd(ebm, ps, st, f_grid)
    Q, P, G = size(f_grid)

    # Choose component if mixture model else use all
    exp_fg = zeros(T, Q, P, G) |> device
    if component_mask !== nothing
        @tullio exp_fg[q, p, g] := (exp(f_grid[q, p, g]) * π_grid[q, g])
        @tullio exp_fg[q, b, g] = exp_fg[q, p, g] * component_mask[q, p, b]
    else
        @tullio exp_fg[q, p, g] := (exp(f_grid[q, p, g]) * π_grid[p, g])
    end

    # CDF by trapezium rule for integration; 1/2 * (u(z_{i-1}) + u(z_i)) * Δx
    exp_fg = exp_fg[:, :, 2:end] + exp_fg[:, :, 1:(end-1)]
    @tullio trapz[q, p, g] := (Δg[p, g] * exp_fg[q, p, g]) / 2
    return trapz, grid, st
end

function get_gausslegendre(
    ebm::Any,
    ps::ComponentArray{T},
    st::NamedTuple,
)::Tuple{AbstractArray{T},AbstractArray{T}} where {T<:half_quant}
    """Get Gauss-Legendre nodes and weights for prior's domain"""

    a, b = minimum(st[Symbol("1")].grid; dims = 2), maximum(st[Symbol("1")].grid; dims = 2)

    no_grid = (
        ebm.fcns_qp[Symbol("1")].spline_string == "FFT" ||
        ebm.fcns_qp[Symbol("1")].spline_string == "Cheby"
    )

    if no_grid
        a = fill(half_quant(first(ebm.fcns_qp[Symbol("1")].grid_range)), size(a)) |> device
        b = fill(half_quant(last(ebm.fcns_qp[Symbol("1")].grid_range)), size(b)) |> device
    end

    nodes = (a + b) ./ 2 .+ (b - a) ./ 2 .* device(ebm.nodes)
    weights = (b - a) ./ 2 .* device(ebm.weights)
    return nodes, weights
end

function gausslegendre_quadrature(
    ebm::Any,
    ps::ComponentArray{T},
    st::NamedTuple;
    ε::T = eps(half_quant),
    component_mask::Union{AbstractArray{T},Nothing} = nothing,
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple} where {T<:half_quant}
    """Gauss-Legendre quadrature for numerical integration"""

    nodes, weights = get_gausslegendre(ebm, ps, st)
    π_nodes =
        ebm.prior_type == "learnable_gaussian" ? ebm.π_pdf(nodes', ps, ε) :
        ebm.π_pdf(nodes, ε)
    π_nodes = ebm.prior_type == "learnable_gaussian" ? π_nodes' : π_nodes

    # Energy function of each component
    nodes, st = prior_fwd(ebm, ps, st, nodes)
    Q, P, G = size(nodes)

    # Choose component if mixture model else use all
    if component_mask !== nothing
        @tullio trapz[q, b, g] :=
            (exp(nodes[q, p, g]) * π_nodes[q, g] * component_mask[q, p, b])
        @tullio trapz[q, b, g] = trapz[q, b, g] * weights[q, g]
        return trapz, nodes, st
    else
        @tullio trapz[q, p, g] := (exp(nodes[q, p, g]) * π_nodes[p, g]) * weights[p, g]
        return trapz, nodes, st
    end
end

@parallel_indices (q, b) function mask_kernel!(
    mask::AbstractArray{T},
    α::AbstractArray{T},
    rand_vals::AbstractArray{T},
    p_size::Int,
)::Nothing where {T<:half_quant}
    idx = p_size + 1
    val = rand_vals[q, b]
    for j = 1:p_size
        if α[q, j] >= val
            idx = j
            break
        end
    end

    # Edge-case: Clamp
    idx = idx > p_size ? p_size : idx

    # One-hot vector for this (q, b)
    for k = 1:p_size
        mask[q, b, k] = (idx == k) ? one(eltype(mask)) : zero(eltype(mask))
    end
    return nothing
end

function choose_component(
    α::AbstractArray{T},
    num_samples::Int,
    q_size::Int,
    p_size::Int;
    seed::Int = 1,
)::Tuple{AbstractArray{T},Int} where {T<:half_quant}
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
    α = cumsum(softmax(full_quant.(α); dims = 2); dims = 2)

    mask = @zeros(q_size, p_size, num_samples)
    @parallel (1:q_size, 1:num_samples) mask_kernel!(mask, α, rand_vals, p_size)
    return mask, seed
end

@parallel_indices (q, p, b) function interp_kernel!(
    z::AbstractArray{T},
    cdf::AbstractArray{T},
    grid::AbstractArray{T},
    rand_vals::AbstractArray{T},
    grid_size::Int,
)::Nothing where {T<:half_quant}
    rv = rand_vals[q, p, b]
    idx = 1

    # Manual searchsortedfirst over cdf[q, p, :]
    for j = 1:grid_size
        if cdf[q, p, j] >= rv
            idx = j
            break
        end
        idx = j + 1
    end

    # Edge cases
    idx = idx == 1 ? 2 : idx
    idx = idx > grid_size ? grid_size : idx

    # Get bounds
    z1, z2 = grid[p, idx-1], grid[p, idx]
    cd1, cd2 = cdf[q, p, idx-1], cdf[q, p, idx]

    # Linear interpolation
    z[q, p, b] = z1 + (z2 - z1) * ((rv - cd1) / (cd2 - cd1))
    return nothing
end

function sample_univariate(
    ebm::Any,
    num_samples::Int,
    ps::ComponentArray{T},
    st::NamedTuple;
    seed::Int = 1,
    ε::T = eps(T),
)::Tuple{AbstractArray{T},NamedTuple,Int} where {T<:half_quant}

    cdf, grid, st = ebm.quad(ebm, ps, st, nothing)
    grid_size = size(grid, 2)
    grid = full_quant.(grid)

    cdf = cat(
        device(zeros(full_quant, ebm.q_size, ebm.p_size, 1)), # Add 0 to start of CDF
        cumsum(full_quant.(cdf); dims = 3), # Cumulative trapezium = CDF
        dims = 3,
    )

    seed, rng = next_rng(seed)
    rand_vals = device(rand(rng, full_quant, 1, ebm.p_size, num_samples))
    rand_vals = rand_vals .* cdf[:, :, end]
    z = @zeros(ebm.q_size, ebm.p_size, num_samples)
    @parallel (1:ebm.q_size, 1:ebm.p_size, 1:num_samples) interp_kernel!(
        z,
        cdf,
        grid,
        rand_vals,
        grid_size,
    )
    return T.(z), st, seed
end

@parallel_indices (q, b) function interp_kernel_mixture!(
    z::AbstractArray{T},
    cdf::AbstractArray{T},
    grid::AbstractArray{T},
    rand_vals::AbstractArray{T},
    grid_size::Int,
)::Nothing where {T<:half_quant}
    rv = rand_vals[q, b]
    idx = 1
    for j = 1:grid_size
        if cdf[q, b, j] >= rv
            idx = j
            break
        end
        idx = j + 1
    end
    idx = idx == 1 ? 2 : idx
    idx = idx > grid_size ? grid_size : idx
    z[q, 1, b] =
        grid[q, idx-1] +
        (grid[q, idx] - grid[q, idx-1]) *
        ((rv - cdf[q, b, idx-1]) / (cdf[q, b, idx] - cdf[q, b, idx-1]))
    return nothing
end

function sample_mixture(
    ebm::Any,
    num_samples::Int,
    ps::ComponentArray{T},
    st::NamedTuple;
    seed::Int = 1,
    ε::T = eps(T),
)::Tuple{AbstractArray{T},NamedTuple,Int} where {T<:half_quant}
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
    mask, seed =
        choose_component(ps[Symbol("α")], num_samples, ebm.q_size, ebm.p_size; seed = seed)

    cdf, grid, st = ebm.quad(ebm, ps, st, mask)
    grid_size = size(grid, 2)
    grid = full_quant.(grid)

    cdf = cat(
        device(zeros(full_quant, ebm.q_size, num_samples, 1)), # Add 0 to start of CDF
        cumsum(full_quant.(cdf); dims = 3), # Cumulative trapezium = CDF
        dims = 3,
    )

    seed, rng = next_rng(seed)
    rand_vals = device(rand(rng, full_quant, ebm.q_size, num_samples))
    rand_vals = rand_vals .* cdf[:, :, end]

    z = @zeros(ebm.q_size, 1, num_samples)
    @parallel (1:ebm.q_size, 1:num_samples) interp_kernel_mixture!(
        z,
        cdf,
        grid,
        rand_vals,
        grid_size,
    )
    return T.(z), st, seed
end

end
