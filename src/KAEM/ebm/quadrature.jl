module Quadrature

export GaussLegendreQuadrature, get_gausslegendre

using LinearAlgebra, Random, ComponentArrays, Accessors

using ..Utils

negative_one = - ones(Float32, 1, 1, 1)

struct GaussLegendreQuadrature <: AbstractQuadrature end

function qfirst_exp_kernel(f, π0)
    return exp.(f) .* PermutedDimsArray(view(π0, :, :, :), (1, 3, 2))
end

function pfirst_exp_kernel(f, π0)
    return exp.(f) .* PermutedDimsArray(view(π0, :, :, :), (3, 1, 2))
end

function apply_mask(exp_fg, component_mask)
    return dropdims(
        sum(
            PermutedDimsArray(
                view(exp_fg, :, :, :, :), (1, 2, 4, 3)
            ) .* component_mask; dims = 2
        ); dims = 2
    )
end

function weight_kernel(trapz, weights)
    return PermutedDimsArray(
        view(weights, :, :, :), (3, 1, 2)
    ) .* trapz
end

function gauss_kernel(
        trapz,
        weights
    )
    return PermutedDimsArray(
        view(weights, :, :, :), (1, 3, 2)
    ) .* trapz
end

function get_gausslegendre(
        ebm,
        st_kan,
    )
    """Get Gauss-Legendre nodes and weights for prior's domain"""
    a, b = st_kan[:a].grid[:, 1], st_kan[:a].grid[:, end]

    if ebm.bool_config.no_grid
        a = st_kan[:a].min
        b = st_kan[:a].max
    end

    nodes, weights = ebm.init_nodes, ebm.init_weights
    nodes = ((a .+ b) ./ 2 .+ (b .- a) ./ 2) * nodes'
    weights = ((b .- a) ./ 2) * weights'
    return nodes, weights
end

function mix_return(nodes, π_nodes, weights, component_mask)
    exp_fg = qfirst_exp_kernel(nodes, π_nodes)
    trapz = apply_mask(exp_fg, component_mask)
    trapz = gauss_kernel(trapz, weights)
    return trapz
end

function univar_return(nodes, π_nodes, weights)
    exp_fg = pfirst_exp_kernel(nodes, π_nodes)
    exp_fg = weight_kernel(exp_fg, weights)
    return exp_fg
end

function (gq::GaussLegendreQuadrature)(
        ebm,
        ps,
        st_kan,
        st_lyrnorm,
        st_quad;
        component_mask = negative_one,
        mix_bool::Bool = false,
    )
    """Gauss-Legendre quadrature for numerical integration"""
    nodes, weights = st_quad.nodes, st_quad.weights
    I, O = first(ebm.fcns_qp).in_dim, first(ebm.fcns_qp).out_dim
    Q, P, S = ebm.q_size, ebm.p_size, ebm.N_quad

    π_nodes = ebm.π_pdf(nodes, ps.dist.π_μ, ps.dist.π_σ)
    π_nodes = ebm.prior_type == "learnable_gaussian" ? π_nodes' : π_nodes

    for i in 1:ebm.depth
        @reset ebm.fcns_qp[i].basis_function.S = S
    end

    @reset ebm.s_size = S

    # Energy function of each component
    nodes, st_lyrnorm_new = ebm(ps, st_kan, st_lyrnorm, nodes)

    # Choose component if mixture model else use all
    result = (
        mix_bool ?
            mix_return(nodes, π_nodes, weights, component_mask) :
            univar_return(nodes, π_nodes, weights)
    )

    return result, st_quad.nodes, st_lyrnorm_new
end

end
