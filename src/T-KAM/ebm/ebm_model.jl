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

include("log_prior_fcns.jl")
include("../kan/univariate_functions.jl")
include("inverse_transform.jl")
include("../../utils.jl")
using .UnivariateFunctions
using .Utils: device, half_quant, full_quant, removeZero, removeNeg, hq, fq, symbol_map
using .LogPriorFCNs
using .InverseTransformSampling

function uniform_pdf(z::AbstractArray{full_quant}, ε::full_quant)::AbstractArray{half_quant}
    return half_quant.((z .>= zero(half_quant)) .* (z .<= one(half_quant))) |> device
end

function gaussian_pdf(z::AbstractArray{full_quant}, ε::full_quant)::AbstractArray{half_quant}
    return half_quant(1 ./ sqrt(2π)) .* exp.(-z .^ 2 ./ 2)
end

function lognormal_pdf(z::AbstractArray{full_quant}, ε::full_quant)::AbstractArray{half_quant}
    return exp.(-(log.(z .+ ε)) .^ 2 ./ 2) ./ (z .* half_quant(sqrt(2π)) .+ ε)
end

function learnable_gaussian_pdf(z::AbstractArray{full_quant}, ps::ComponentArray{full_quant}, ε::full_quant)::AbstractArray{half_quant}
    return one(half_quant) ./ (abs.(ps.dist.π_σ .* half_quant(sqrt(2π)) .+ ε) .* exp.(-(z .- ps.dist.π_μ .^ 2) ./ (2 .* (ps.dist.π_σ .^ 2) .+ ε)))
end

function ebm_pdf(z::AbstractArray{full_quant}, ε::full_quant)::AbstractArray{half_quant}
    return ones(half_quant, size(z)) .- ε |> device
end

const prior_pdf = Dict(
    "uniform" => uniform_pdf,
    "gaussian" => gaussian_pdf,
    "lognormal" => lognormal_pdf,
    "ebm" => ebm_pdf,
    "learnable_gaussian" => learnable_gaussian_pdf,
)

const quad_map =
    Dict("gausslegendre" => gausslegendre_quadrature, "trapezium" => trapezium_quadrature)

struct EbmModel{T<:half_quant} <: Lux.AbstractLuxLayer
    fcns_qp::Tuple
    layernorms::Tuple
    layernorm_bool::Bool
    depth::Int
    prior_type::AbstractString
    π_pdf::Function
    sample_z::Function
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
    layernorm_bool = parse(Bool, retrieve(conf, "EbmModel", "layer_norm"))
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

    sample_function =
        (m, n, p, s, rng) -> begin
            if mixture_model
                sample_mixture(m.prior, n, p.ebm, Lux.testmode(s.ebm); rng = rng, ε = eps)
            else
                sample_univariate(m.prior, n, p.ebm, Lux.testmode(s.ebm); rng = rng, ε = eps)
            end
        end

    fcns_temp = []
    layernorms_temp = []
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

        push!(fcns_temp, func)

        if (layernorm_bool && i < length(widths)-1)
            push!(layernorms_temp, Lux.LayerNorm(widths[i+1]))
        end

    end

    functions = ntuple(i -> fcns_temp[i], length(widths)-1)
    layernorms = ()
    if layernorm_bool && length(layernorms_temp) > 0
        layernorms = ntuple(i -> layernorms_temp[i], length(widths)-1)
    end

    ula = length(widths) > 2
    contrastive_div =
        parse(Bool, retrieve(conf, "TRAINING", "contrastive_divergence_training")) && !ula

    quad_type = retrieve(conf, "EbmModel", "quadrature_method")
    quad_fcn = get(quad_map, quad_type, gausslegendre_quadrature)
    quadrature_method = (m, p, s, mask) -> quad_fcn(m, p, s; ε = eps, component_mask = mask)

    N_quad = parse(Int, retrieve(conf, "EbmModel", "GaussQuad_nodes"))
    nodes, weights = gausslegendre(N_quad)
    nodes = repeat(nodes', first(widths), 1) .|> half_quant
    weights = half_quant.(weights)'

    lp_fcn = begin
        if mixture_model && !ula
            log_prior_mix
        elseif ula
            log_prior_ula
        else
            log_prior_univar
        end
    end

    return EbmModel(
        functions,
        layernorms,
        layernorm_bool,
        length(widths)-1,
        prior_type,
        get(prior_pdf, prior_type, (z, ε) -> ones(half_quant, size(z))),
        sample_function,
        P,
        Q,
        quadrature_method,
        N_quad,
        nodes,
        weights,
        contrastive_div,
        quad_type,
        ula,
        lp_fcn,
        mixture_model,
        reg,
    )
end

function Lux.initialparameters(rng::AbstractRNG, prior::EbmModel{T}) where {T<:half_quant}
    # fcn_ps = ntuple(i -> Lux.initialparameters(rng, prior.fcns_qp[i]), prior.depth)
    fcn_ps = NamedTuple(symbol_map[i] => Lux.initialparameters(rng, prior.fcns_qp[i]) for i in 1:prior.depth)
    layernorm_ps = NamedTuple()
    if prior.layernorm_bool && length(prior.layernorms) > 0
        layernorm_ps = NamedTuple(symbol_map[i] => Lux.initialparameters(rng, prior.layernorms[i]) for i in 1:prior.depth-1)
    end

    prior_ps = (
    π_μ = prior.prior_type == "learnable_gaussian" ? zeros(half_quant, 1, prior.p_size) : nothing,
    π_σ = prior.prior_type == "learnable_gaussian" ? ones(half_quant, 1, prior.p_size) : nothing,
    α   = prior.mixture_model ? glorot_uniform(rng, full_quant, prior.q_size, prior.p_size) : nothing,
    )
   
    return(
        fcn = fcn_ps,
        dist = prior_ps,
        layernorm = layernorm_ps,
    )
end

function Lux.initialstates(rng::AbstractRNG, prior::EbmModel{T}) where {T<:half_quant}
    fcn_st = NamedTuple(symbol_map[i] => Lux.initialstates(rng, prior.fcns_qp[i]) for i in 1:prior.depth)
    layernorm_st = NamedTuple()
    if prior.layernorm_bool && length(prior.layernorms) > 0
        layernorm_st = NamedTuple(symbol_map[i] => Lux.initialstates(rng, prior.layernorms[i]) |> hq for i in 1:prior.depth-1)
    end

    return (
        fcn = fcn_st,
        layernorm = layernorm_st,
    )
end

end
