module Quadrature

export TrapeziumQuadrature, GaussLegendreQuadrature

using CUDA, KernelAbstractions, LinearAlgebra, Random, Lux, LuxCUDA, ComponentArrays, ParallelStencil

using ..Utils

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, half_quant, 3)
else
    @init_parallel_stencil(Threads, half_quant, 3)
end

negative_one = pu([-one(half_quant)])

struct TrapeziumQuadrature <: Lux.AbstractLuxLayer end

struct GaussLegendreQuadrature <: Lux.AbstractLuxLayer end

@parallel_indices (q, p, g) function qfirst_exp_kernel!(
    exp_fg::AbstractArray{T},
    f::AbstractArray{T},
    π0::AbstractArray{T},
)::Nothing where {T<:half_quant}
    exp_fg[q, p, g] = exp(f[q, p, g]) * π0[q, g]
    return nothing
end

@parallel_indices (q, p, g) function pfirst_exp_kernel!(
    exp_fg::AbstractArray{T},
    f::AbstractArray{T},
    π0::AbstractArray{T},
)::Nothing where {T<:half_quant}
    exp_fg[q, p, g] = exp(f[q, p, g]) * π0[p, g]
    return nothing
end

@parallel_indices (q, b, g) function apply_mask!(
    trapz::AbstractArray{T},
    exp_fg::AbstractArray{T},
    component_mask::AbstractArray{T},
)::Nothing where {T<:half_quant}
    acc = zero(T)
    for p = 1:size(component_mask, 2)
        acc += exp_fg[q, p, g] * component_mask[q, p, b]
    end
    trapz[q, b, g] = acc
    return nothing
end

@parallel_indices (q, p, g) function weight_kernel!(
    trapz::AbstractArray{T},
    weight::AbstractArray{T},
)::Nothing where {T<:half_quant}
    trapz[q, p, g] = weight[p, g] * trapz[q, p, g]
    return nothing
end

@parallel_indices (q, b, g) function gauss_kernel!(
    trapz::AbstractArray{T},
    weight::AbstractArray{T},
)::Nothing where {T<:half_quant}
    trapz[q, b, g] = weight[q, g] * trapz[q, b, g]
    return nothing
end

function (tq::TrapeziumQuadrature)(
    ebm::Lux.AbstractLuxLayer,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple;
    component_mask::AbstractArray{T} = negative_one,
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple} where {T<:half_quant}
    """Trapezoidal rule for numerical integration: 1/2 * (u(z_{i-1}) + u(z_i)) * Δx"""

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
    exp_fg = @zeros(Q, P, G)
    if !any(component_mask .< zero(T))
        B = size(component_mask, 3)
        trapz = @zeros(Q, B, G)
        @parallel (1:Q, 1:P, 1:G) qfirst_exp_kernel!(exp_fg, f_grid, π_grid)
        @parallel (1:Q, 1:B, 1:G) apply_mask!(trapz, exp_fg, component_mask)
        trapz = trapz[:, :, 2:end] + trapz[:, :, 1:(end-1)]
        @parallel (1:Q, 1:B, 1:(G-1)) weight_kernel!(trapz, Δg)
        @. trapz = trapz / 2
        return trapz, st_kan[:a].grid, st_lyrnorm_new
    else
        @parallel (1:Q, 1:P, 1:G) pfirst_exp_kernel!(exp_fg, f_grid, π_grid)
        trapz = exp_fg[:, :, 2:end] + exp_fg[:, :, 1:(end-1)]
        @parallel (1:Q, 1:P, 1:(G-1)) weight_kernel!(trapz, Δg)
        @. trapz = trapz / 2
        return trapz, st_kan[:a].grid, st_lyrnorm_new
    end
end

function get_gausslegendre(
    ebm::Lux.AbstractLuxLayer,
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
    ebm::Lux.AbstractLuxLayer,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple;
    component_mask::AbstractArray{T} = negative_one,
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
    exp_fg = @zeros(Q, P, G)
    if !any(component_mask .< zero(T))
        B = size(component_mask, 3)
        trapz = @zeros(Q, B, G)
        @parallel (1:Q, 1:P, 1:G) qfirst_exp_kernel!(exp_fg, nodes, π_nodes)
        @parallel (1:Q, 1:B, 1:G) apply_mask!(trapz, exp_fg, component_mask)
        @parallel (1:Q, 1:B, 1:G) gauss_kernel!(trapz, weights)
        return trapz, grid, st_lyrnorm_new
    else
        trapz = @zeros(Q, P, G)
        @parallel (1:Q, 1:P, 1:G) pfirst_exp_kernel!(exp_fg, nodes, π_nodes)
        @parallel (1:Q, 1:P, 1:G) weight_kernel!(trapz, weights)
        return trapz, grid, st_lyrnorm_new
    end
end

end
