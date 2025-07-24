module EBM_Model

export EbmModel, init_EbmModel, log_prior

using CUDA, KernelAbstractions, FastGaussQuadrature
using ConfParser,
    Random,
    Distributions,
    Lux,
    Accessors,
    LuxCUDA,
    Statistics,
    LinearAlgebra,
    ComponentArrays

using ..Utils

include("../kan/univariate_functions.jl")
include("inverse_transform.jl")
include("ref_priors.jl")
using .UnivariateFunctions: univariate_function, init_function
using .InverseTransformSampling
using .RefPriors: prior_pdf

const quad_map =
    Dict("gausslegendre" => gausslegendre_quadrature, "trapezium" => trapezium_quadrature)

struct EbmModel{T<:half_quant,U<:full_quant} <: Lux.AbstractLuxLayer
    fcns_qp::Vector{univariate_function{T,U}}
    layernorms::Vector{Lux.LayerNorm}
    layernorm_bool::Bool
    depth::Int
    prior_type::AbstractString
    π_pdf!::Any
    p_size::Int
    q_size::Int
    quad::Function
    N_quad::Int
    nodes::AbstractArray{T}
    weights::AbstractArray{T}
    contrastive_div::Bool
    quad_type::AbstractString
    ula::Bool
    lp_fcn::Function
    mixture_model::Bool
    λ::half_quant
end

function init_EbmModel(conf::ConfParse; rng::AbstractRNG = Random.default_rng())
    widths = (
        try
            parse.(Int, retrieve(conf, "EbmModel", "layer_widths"))
        catch
            parse.(Int, split(retrieve(conf, "EbmModel", "layer_widths"), ","))
        end
    )

    spline_degree = parse(Int, retrieve(conf, "EbmModel", "spline_degree"))
    layernorm_bool = parse(Bool, retrieve(conf, "EbmModel", "layernorm"))
    base_activation = retrieve(conf, "EbmModel", "base_activation")
    spline_function = retrieve(conf, "EbmModel", "spline_function")
    grid_size = parse(Int, retrieve(conf, "EbmModel", "grid_size"))
    grid_update_ratio = parse(half_quant, retrieve(conf, "EbmModel", "grid_update_ratio"))
    ε_scale = parse(half_quant, retrieve(conf, "EbmModel", "ε_scale"))
    μ_scale = parse(full_quant, retrieve(conf, "EbmModel", "μ_scale"))
    σ_base = parse(full_quant, retrieve(conf, "EbmModel", "σ_base"))
    σ_spline = parse(full_quant, retrieve(conf, "EbmModel", "σ_spline"))
    init_τ = parse(full_quant, retrieve(conf, "EbmModel", "init_τ"))
    τ_trainable = parse(Bool, retrieve(conf, "EbmModel", "τ_trainable"))
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
    τ_trainable = spline_function == "B-spline" ? false : τ_trainable
    reg = parse(half_quant, retrieve(conf, "EbmModel", "λ_reg"))

    P, Q = first(widths), last(widths)

    grid_range = parse.(half_quant, retrieve(conf, "EbmModel", "grid_range"))
    prior_type = retrieve(conf, "EbmModel", "π_0")
    mixture_model = parse(Bool, retrieve(conf, "EbmModel", "mixture_model"))
    widths = mixture_model ? reverse(widths) : widths

    grid_range_first = Dict(
        "ebm" => grid_range,
        "learnable_gaussian" => grid_range,
        "lognormal" => [0, 4] .|> half_quant,
        "gaussian" => [-1, 1] .|> half_quant,
        "uniform" => [0, 1] .|> half_quant,
    )[prior_type]

    eps = parse(half_quant, retrieve(conf, "TRAINING", "eps"))

    functions = []
    layernorms = []
    for i in eachindex(widths[1:(end-1)])
        base_scale = (
            μ_scale * (one(full_quant) / √(full_quant(widths[i]))) .+
            σ_base .* (
                randn(rng, full_quant, widths[i], widths[i+1]) .* full_quant(2) .-
                one(full_quant)
            ) .* (one(full_quant) / √(full_quant(widths[i])))
        )

        grid_range_i = i == 1 ? grid_range_first : grid_range

        func = init_function(
            widths[i],
            widths[i+1];
            spline_degree = spline_degree,
            base_activation = base_activation,
            spline_function = spline_function,
            grid_size = grid_size,
            grid_update_ratio = grid_update_ratio,
            grid_range = Tuple(grid_range_i),
            ε_scale = ε_scale,
            σ_base = base_scale,
            σ_spline = σ_spline,
            init_τ = init_τ,
            τ_trainable = τ_trainable,
        )

        push!(functions, func)

        if (layernorm_bool && i < length(widths)-1)
            push!(layernorms, Lux.LayerNorm(widths[i+1]))
        end
    end

    ula = length(widths) > 2
    contrastive_div =
        parse(Bool, retrieve(conf, "TRAINING", "contrastive_divergence_training")) && !ula

    quad_type = retrieve(conf, "EbmModel", "quadrature_method")
    quad_fcn = get(quad_map, quad_type, gausslegendre_quadrature)
    quadrature_method =
        (m, p, sk, sl, mask) -> quad_fcn(m, p, sk, sl; ε = eps, component_mask = mask)

    N_quad = parse(Int, retrieve(conf, "EbmModel", "GaussQuad_nodes"))
    nodes, weights = gausslegendre(N_quad)
    nodes = repeat(nodes', first(widths), 1) .|> half_quant
    weights = half_quant.(weights')



    ref_initializer = get(prior_pdf, prior_type, prior_pdf["uniform"])

    return EbmModel(
        functions,
        layernorms,
        layernorm_bool,
        length(widths)-1,
        prior_type,
        ref_initializer(eps),
        P,
        Q,
        quadrature_method,
        N_quad,
        nodes,
        weights,
        contrastive_div,
        quad_type,
        ula,
        identity,
        mixture_model,
        reg,
    )
end

function (ebm::EbmModel{T})(
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple,
    z::AbstractArray{T},
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
    """
    Forward pass through the ebm-prior, returning the energy function.

    Args:
        ebm: The ebm-prior.
        ps: The parameters of the ebm-prior.
        st: The states of the ebm-prior.
        z: The component-wise latent samples to evaulate the measure on, (q, num_samples) or (p, num_samples)

    Returns:
        f: The energy function, (num_samples,) or (q, p, num_samples)
        st: The updated states of the ebm-prior.
    """

    mid_size = !ebm.mixture_model ? ebm.p_size : ebm.q_size

    for i = 1:ebm.depth
        z = ebm.fcns_qp[i](z, ps.fcn[symbol_map[i]], st_kan[symbol_map[i]])

        z =
            (i == 1 && !ebm.ula) ? reshape(z, size(z, 2), mid_size*size(z, 3)) :
            dropdims(sum(z, dims = 1); dims = 1)

        z, st_lyrnorm_new =
            (ebm.layernorm_bool && i < ebm.depth) ?
            Lux.apply(ebm.layernorms[i], z, ps.layernorm[i], st_lyrnorm[i]) :
            (z, st_lyrnorm)

        (ebm.layernorm_bool && i < ebm.depth) && @reset st_lyrnorm[i] = st_lyrnorm_new
    end

    z = ebm.ula ? z : reshape(z, ebm.q_size, ebm.p_size, :)
    return z, st_lyrnorm
end

function Lux.initialparameters(rng::AbstractRNG, prior::EbmModel{T}) where {T<:half_quant}
    fcn_ps = NamedTuple(
        symbol_map[i] => Lux.initialparameters(rng, prior.fcns_qp[i]) for i = 1:prior.depth
    )
    layernorm_ps = (a = zero(T))
    if prior.layernorm_bool && length(prior.layernorms) > 0
        layernorm_ps = NamedTuple(
            symbol_map[i] => Lux.initialparameters(rng, prior.layernorms[i]) for
            i = 1:(prior.depth-1)
        )
    end

    prior_ps = (
        π_μ = prior.prior_type == "learnable_gaussian" ?
              zeros(half_quant, prior.p_size) : [zero(T)],
        π_σ = prior.prior_type == "learnable_gaussian" ?
              ones(half_quant, prior.p_size) : [zero(T)],
        α = prior.mixture_model ?
            glorot_uniform(rng, full_quant, prior.q_size, prior.p_size) : [zero(T)],
    )

    return (fcn = fcn_ps, dist = prior_ps, layernorm = layernorm_ps)
end

function Lux.initialstates(rng::AbstractRNG, prior::EbmModel{T}) where {T<:half_quant}
    fcn_st = NamedTuple(
        symbol_map[i] => Lux.initialstates(rng, prior.fcns_qp[i]) for i = 1:prior.depth
    )
    st_lyrnorm = (a = zero(T), b = zero(T))
    if prior.layernorm_bool && length(prior.layernorms) > 0
        st_lyrnorm = NamedTuple(
            symbol_map[i] => Lux.initialstates(rng, prior.layernorms[i]) |> hq for
            i = 1:(prior.depth-1)
        )
    end

    # KAN states are meant to be a ComponentArray - return separately
    return fcn_st, st_lyrnorm
end

end
