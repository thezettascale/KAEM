module EBM_Model

export EbmModel, init_EbmModel, log_prior

using CUDA, KernelAbstractions, FastGaussQuadrature
using ConfParser, Random, Distributions, Lux, Accessors, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays
using ChainRules: @ignore_derivatives

include("log_prior_fcns.jl")
include("../kan/univariate_functions.jl")
include("inverse_transform.jl")
include("../../utils.jl")
using .univariate_functions
using .Utils: device, next_rng, half_quant, full_quant, removeZero, removeNeg, hq, fq
using .LogPriorFCNs
using .InverseTransformSampling

prior_pdf = Dict(
    "uniform" => (z, ε) -> half_quant.(0 .<= z .<= 1) |> device,
    "gaussian" => (z, ε) -> half_quant(1 ./ sqrt(2π)) .* exp.(-z.^2 ./ 2),
    "lognormal" => (z, ε) -> exp.(-(log.(z .+ ε)).^2 ./ 2) ./ (z .* half_quant(sqrt(2π)) .+ ε),
    "ebm" => (z, ε) -> ones(half_quant, size(z)) .- ε |> device,
    "learnable_gaussian" => (z, ps, ε) -> (
        one(half_quant) ./ (abs.(ps[Symbol("π_σ")]) .* half_quant(sqrt(2π)) .+ ε) 
    .* exp.(-(z .- ps[Symbol("π_μ")].^2) ./ (2 .* (ps[Symbol("π_σ")].^2) .+ ε))
    )
)

struct EbmModel{T<:half_quant} <: Lux.AbstractLuxLayer
    fcns_qp::NamedTuple
    layernorm::Bool
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

function init_EbmModel(
    conf::ConfParse;
    prior_seed::Int=1,
    )
    widths = (
        try 
            parse.(Int, retrieve(conf, "EbmModel", "layer_widths"))
        catch
            parse.(Int, split(retrieve(conf, "EbmModel", "layer_widths"), ","))
        end
    )

    spline_degree = parse(Int, retrieve(conf, "EbmModel", "spline_degree"))
    layernorm = parse(Bool, retrieve(conf, "EbmModel", "layer_norm"))
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
        "lognormal" => [0,4] .|> half_quant,
        "gaussian" => [-1,1] .|> half_quant,
        "uniform" => [0,1] .|> half_quant,
    )[prior_type]

    eps = parse(half_quant, retrieve(conf, "TRAINING", "eps"))
    
    sample_function = (m, n, p, s, seed) -> begin
        if mixture_model
            @ignore_derivatives sample_mixture(m.prior, n, p.ebm, Lux.testmode(s.ebm); seed=seed, ε=eps)
        else
            @ignore_derivatives sample_univariate(m.prior, n, p.ebm, Lux.testmode(s.ebm); seed=seed, ε=eps)
        end
    end    

    functions = NamedTuple()
    for i in eachindex(widths[1:end-1])
        prior_seed, rng = next_rng(prior_seed)
        base_scale = (μ_scale * (one(full_quant) / √(full_quant(widths[i])))
        .+ σ_base .* (randn(rng, full_quant, widths[i], widths[i+1]) .* full_quant(2) .- one(full_quant)) .* (one(full_quant) / √(full_quant(widths[i]))))

        grid_range_i = i == 1 ? grid_range_first : grid_range

        func = init_function(
        widths[i],
        widths[i+1];
        spline_degree=spline_degree,
        base_activation=base_activation,
        spline_function=spline_function,
        grid_size=grid_size,
        grid_update_ratio=grid_update_ratio,
        grid_range=Tuple(grid_range_i),
        ε_scale=ε_scale,
        σ_base=base_scale,
        σ_spline=σ_spline,
        init_τ=init_τ,
        τ_trainable=τ_trainable,
        )

        @reset functions[Symbol("$i")] = func

        if (layernorm && i < length(widths)-1)
            @reset functions[Symbol("ln_$i")] = Lux.LayerNorm(widths[i+1]) 
        end

    end

    ula = length(widths) > 2 
    contrastive_div = parse(Bool, retrieve(conf, "TRAINING", "contrastive_divergence_training")) && !ula

    quad_type = retrieve(conf, "EbmModel", "quadrature_method")
    quadrature_method = Dict(
        "gausslegendre" => (m, p, s, mask) -> gausslegendre_quadrature(m, p, s; ε=eps, component_mask=mask),
        "trapezium" => (m, p, s, mask) -> trapezium_quadrature(m, p, s; ε=eps, component_mask=mask),
    )[quad_type]

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
        layernorm, 
        length(widths)-1, 
        prior_type, 
        prior_pdf[prior_type], 
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
        reg
        )
end

function Lux.initialparameters(rng::AbstractRNG, prior::EbmModel)
    ps = NamedTuple(Symbol("$i") => Lux.initialparameters(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)

    if prior.layernorm 
        for i in 1:prior.depth-1
            @reset ps[Symbol("ln_$i")] = Lux.initialparameters(rng, prior.fcns_qp[Symbol("ln_$i")]) 
        end
    end

    if prior.prior_type == "learnable_gaussian"
        @reset ps[Symbol("π_μ")] = zeros(half_quant, 1, prior.p_size)
        @reset ps[Symbol("π_σ")] = ones(half_quant, 1, prior.p_size)
    end

    if prior.mixture_model
        @reset ps[Symbol("α")] = glorot_uniform(rng, full_quant, prior.q_size, prior.p_size)
    end
    
    return ps 
end
 
function Lux.initialstates(rng::AbstractRNG, prior::EbmModel)
    st = NamedTuple(Symbol("$i") => Lux.initialstates(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)

    if prior.layernorm 
        for i in 1:prior.depth-1
            @reset st[Symbol("ln_$i")] = Lux.initialstates(rng, prior.fcns_qp[Symbol("ln_$i")]) |> hq
        end
    end

    return st 
end

end