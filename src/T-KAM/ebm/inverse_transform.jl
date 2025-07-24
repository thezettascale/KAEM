
module InverseTransformSampling

export sample_univariate, sample_mixture, GaussLegendreQuadrature, TrapeziumQuadrature

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

using ..Utils

include("mixture_selection.jl")
using .MixtureChoice: choose_component

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, half_quant, 3)
else
    @init_parallel_stencil(Threads, half_quant, 3)
end

struct TrapeziumQuadrature <: Lux.AbstractLuxLayer end

struct GaussLegendreQuadrature <: Lux.AbstractLuxLayer end

function (tq::TrapeziumQuadrature)(
    ebm,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple;
    component_mask::Union{AbstractArray{<:half_quant},Nothing} = nothing,
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple} where {T<:half_quant}
    """Trapezoidal rule for numerical integration"""

    # Evaluate prior on grid [0,1]
    f_grid = st_kan[:a].grid
    Δg = f_grid[:, 2:end] - f_grid[:, 1:(end-1)]

    I, O = size(f_grid)
    π_grid = @zeros(I, O, 1)
    ebm.π_pdf!(π_grid, f_grid[:, :, :], ps.dist.π_μ, ps.dist.π_σ)
    π_grid =
        ebm.prior_type == "learnable_gaussian" ? dropdims(π_grid, dims = 3)' :
        dropdims(π_grid, dims = 3)

    # Energy function of each component
    f_grid, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, f_grid)
    Q, P, G = size(f_grid)

    # Choose component if mixture model else use all
    exp_fg = zeros(T, Q, P, G) |> pu
    if component_mask !== nothing
        @tullio exp_fg[q, p, g] := (exp(f_grid[q, p, g]) * π_grid[q, g])
        @tullio exp_fg[q, b, g] = exp_fg[q, p, g] * component_mask[q, p, b]
    else
        @tullio exp_fg[q, p, g] := (exp(f_grid[q, p, g]) * π_grid[p, g])
    end

    # CDF by trapezium rule for integration; 1/2 * (u(z_{i-1}) + u(z_i)) * Δx
    exp_fg = exp_fg[:, :, 2:end] + exp_fg[:, :, 1:(end-1)]
    @tullio trapz[q, p, g] := (Δg[p, g] * exp_fg[q, p, g]) / 2
    return trapz, st_kan[:a].grid, st_lyrnorm_new
end

function get_gausslegendre(
    ebm,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
)::Tuple{AbstractArray{T},AbstractArray{T}} where {T<:half_quant}
    """Get Gauss-Legendre nodes and weights for prior's domain"""

    a, b = minimum(st_kan[:a].grid; dims = 2), maximum(st_kan[:a].grid; dims = 2)

    no_grid =
        (ebm.fcns_qp[1].spline_string == "FFT" || ebm.fcns_qp[1].spline_string == "Cheby")

    if no_grid
        a = fill(half_quant(first(ebm.fcns_qp[1].grid_range)), size(a)) |> pu
        b = fill(half_quant(last(ebm.fcns_qp[1].grid_range)), size(b)) |> pu
    end

    nodes, weights = pu(ebm.nodes), pu(ebm.weights)
    @. nodes = (a + b) / 2 + (b - a) / 2 * nodes
    return nodes, (b - a) ./ 2 .* weights
end

function (gq::GaussLegendreQuadrature)(
    ebm,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple;
    component_mask::Union{AbstractArray{T},Nothing} = nothing,
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple} where {T<:half_quant}
    """Gauss-Legendre quadrature for numerical integration"""

    nodes, weights = get_gausslegendre(ebm, ps, st_kan)
    grid = nodes

    I, O = size(nodes)
    π_nodes = @zeros(I, O, 1)
    ebm.π_pdf!(π_nodes, nodes[:, :, :], ps.dist.π_μ, ps.dist.π_σ)
    π_nodes =
        ebm.prior_type == "learnable_gaussian" ? dropdims(π_nodes, dims = 3)' :
        dropdims(π_nodes, dims = 3)

    # Energy function of each component
    nodes, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, nodes)
    Q, P, G = size(nodes)

    # Choose component if mixture model else use all
    if component_mask !== nothing
        @tullio trapz[q, b, g] :=
            (exp(nodes[q, p, g]) * π_nodes[q, g] * component_mask[q, p, b])
        @tullio trapz[q, b, g] = trapz[q, b, g] * weights[q, g]
        return trapz, grid, st_lyrnorm_new
    else
        @tullio trapz[q, p, g] := (exp(nodes[q, p, g]) * π_nodes[p, g]) * weights[p, g]
        return trapz, grid, st_lyrnorm_new
    end
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

    # Manual searchsortedfirst over cdf[q, p, :] - potential thread divergence on GPU
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
    ebm,
    num_samples::Int,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple;
    rng::AbstractRNG = Random.default_rng(),
    ε::T = eps(T),
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}

    cdf, grid, st_lyrnorm_new = ebm.quad(ebm, ps, st_kan, st_lyrnorm, nothing)
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
    z::AbstractArray{T},
    cdf::AbstractArray{T},
    grid::AbstractArray{T},
    rand_vals::AbstractArray{T},
    grid_size::Int,
)::Nothing where {T<:half_quant}
    rv = rand_vals[q, b]
    idx = 1

    # Manual searchsortedfirst over cdf[q, b, :] - potential thread divergence on GPU
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
    ebm,
    num_samples::Int,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple;
    rng::AbstractRNG = Random.default_rng(),
    ε::T = eps(T),
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
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
    mask = choose_component(ps.α, num_samples, ebm.q_size, ebm.p_size; rng = rng)

    cdf, grid, st_lyrnorm_new = ebm.quad(ebm, ps, st_kan, st_lyrnorm, mask)
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
