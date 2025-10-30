module Quadrature

export TrapeziumQuadrature, GaussLegendreQuadrature

using CUDA, KernelAbstractions, LinearAlgebra, Random, Lux, LuxCUDA, ComponentArrays, Tullio

using ..Utils

negative_one = - ones(half_quant, 1, 1, 1) |> pu

struct TrapeziumQuadrature <: AbstractQuadrature end

struct GaussLegendreQuadrature <: AbstractQuadrature end

function qfirst_exp_kernel(
    f::AbstractArray{T,3},
    π0::AbstractArray{T,2},
)::AbstractArray{T,3} where {T<:half_quant}
    return @tullio exp_fg[q, p, g] := exp(f[q, p, g]) * π0[q, g]
end

function pfirst_exp_kernel(
    f::AbstractArray{T,3},
    π0::AbstractArray{T,2},
)::AbstractArray{T,3} where {T<:half_quant}
    return @tullio exp_fg[q, p, g] := exp(f[q, p, g]) * π0[p, g]
end

function apply_mask(
    exp_fg::AbstractArray{T,3},
    component_mask::AbstractArray{T,3},
)::AbstractArray{T,3} where {T<:half_quant}
    return @tullio trapz[q, b, g] := exp_fg[q, p, g] * component_mask[q, p, b]
end

function weight_kernel(
    trapz::AbstractArray{T,3},
    weight::AbstractArray{T,2},
)::AbstractArray{T,3} where {T<:half_quant}
    return @tullio trapz_weighted[q, p, g] := weight[p, g] * trapz[q, p, g]
end

function gauss_kernel(
    trapz::AbstractArray{T,3},
    weight::AbstractArray{T,2},
)::AbstractArray{T,3} where {T<:half_quant}
    return @tullio trapz_weighted[q, b, g] := weight[q, g] * trapz[q, b, g]
end

function (tq::TrapeziumQuadrature)(
    ebm::Lux.AbstractLuxLayer,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple;
    component_mask::AbstractArray{T,3} = negative_one,
)::Tuple{AbstractArray{T,3},AbstractArray{T,2},NamedTuple} where {T<:half_quant}
    """Trapezoidal rule for numerical integration: 1/2 * (u(z_{i-1}) + u(z_i)) * Δx"""

    # Evaluate prior on grid [0,1]
    f_grid = st_kan[:a].grid
    Δg = f_grid[:, 2:end] - f_grid[:, 1:(end-1)]

    I, O = size(f_grid)
    π_grid = ebm.π_pdf(f_grid[:, :, :], ps.dist.π_μ, ps.dist.π_σ)
    π_grid =
        ebm.prior_type == "learnable_gaussian" ? dropdims(π_grid, dims = 3)' :
        dropdims(π_grid, dims = 3)

    # Energy function of each component
    f_grid, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, f_grid)
    Q, P, G = size(f_grid)

    # Choose component if mixture model else use all
    if !any(component_mask .< zero(T))
        B = size(component_mask, 3)
        exp_fg = qfirst_exp_kernel(f_grid, π_grid)
        trapz = apply_mask(exp_fg, component_mask)
        trapz = trapz[:, :, 2:end] + trapz[:, :, 1:(end-1)]
        trapz = weight_kernel(trapz, Δg)
        return trapz ./ 2, st_kan[:a].grid, st_lyrnorm_new
    else
        exp_fg = pfirst_exp_kernel(f_grid, π_grid)
        trapz = exp_fg[:, :, 2:end] + exp_fg[:, :, 1:(end-1)]
        trapz = weight_kernel(trapz, Δg)
        return trapz ./ 2, st_kan[:a].grid, st_lyrnorm_new
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
        a = fill(half_quant(first(ebm.prior_domain)), size(a)) |> pu
        b = fill(half_quant(last(ebm.prior_domain)), size(b)) |> pu
    end

    nodes, weights = pu(ebm.nodes), pu(ebm.weights)
    return ((a + b) / 2 + (b - a) / 2) .* nodes, (b - a) ./ 2 .* weights
end

function (gq::GaussLegendreQuadrature)(
    ebm::Lux.AbstractLuxLayer,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple;
    component_mask::AbstractArray{T,3} = negative_one,
)::Tuple{AbstractArray{T,3},AbstractArray{T,2},NamedTuple} where {T<:half_quant}
    """Gauss-Legendre quadrature for numerical integration"""

    nodes, weights = get_gausslegendre(ebm, ps, st_kan)
    grid = nodes

    I, O = size(nodes)
    π_nodes = ebm.π_pdf(nodes[:, :, :], ps.dist.π_μ, ps.dist.π_σ)
    π_nodes =
        ebm.prior_type == "learnable_gaussian" ? dropdims(π_nodes, dims = 3)' :
        dropdims(π_nodes, dims = 3)

    # Energy function of each component
    nodes, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, nodes)
    Q, P, G = size(nodes)

    # Choose component if mixture model else use all
    if !any(component_mask .< zero(T))
        B = size(component_mask, 3)
        exp_fg = qfirst_exp_kernel(nodes, π_nodes)
        trapz = apply_mask(exp_fg, component_mask)
        trapz = gauss_kernel(trapz, weights)
        return trapz, grid, st_lyrnorm_new
    else
        exp_fg = pfirst_exp_kernel(nodes, π_nodes)
        exp_fg = weight_kernel(exp_fg, weights)
        return exp_fg, grid, st_lyrnorm_new
    end
end

end
