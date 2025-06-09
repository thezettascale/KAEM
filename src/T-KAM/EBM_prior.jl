module ebm_ebm_prior

export ebm_prior, init_ebm_prior, log_prior

using CUDA, KernelAbstractions, Tullio, FastGaussQuadrature
using ConfParser, Random, Distributions, Lux, Accessors, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays
using ChainRules: @ignore_derivatives

include("log_prior_fcns.jl")
include("univariate_functions.jl")
include("sampling/inverse_transform.jl")
include("../utils.jl")
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

struct ebm_prior{T<:half_quant} <: Lux.AbstractLuxLayer
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

function trapezium_quadrature(
    ebm, 
    ps, 
    st; 
    ε::T=eps(half_quant),
    component_mask::Union{AbstractArray{<:half_quant}, Nothing}=nothing
    ) where {T<:half_quant}
    """Trapezoidal rule for numerical integration"""

    # Evaluate prior on grid [0,1]
    f_grid = st[Symbol("1")].grid
    grid = f_grid |> cpu_device() 
    Δg = f_grid[:, 2:end] - f_grid[:, 1:end-1] 
    
    π_grid = ebm.prior_type == "learnable_gaussian" ? ebm.π_pdf(f_grid', ps, ε) : ebm.π_pdf(f_grid, ε)
    π_grid = ebm.prior_type == "learnable_gaussian" ? π_grid' : π_grid
    grid_size = size(f_grid, 2)

    # Energy function of each component
    f_grid, st = prior_fwd(ebm, ps, st, f_grid)
   
    exp_fg = zeros(T, ebm.q_size, ebm.p_size, grid_size) |> device
    if component_mask !== nothing
        @tullio exp_fg[q, p, g] := (exp(f_grid[q, p, g]) * π_grid[q, g])
        @tullio exp_fg[q, b, g] = exp_fg[q, p, g] * component_mask[q, p, b] 
    else
        @tullio exp_fg[q, p, g] := (exp(f_grid[q, p, g]) * π_grid[p, g])
    end

    # CDF evaluated by trapezium rule for integration; 1/2 * (u(z_{i-1}) + u(z_i)) * Δx
    exp_fg = exp_fg[:, :, 2:end] + exp_fg[:, :, 1:end-1] 
    @tullio trapz[q, p, g] := (Δg[p, g] * exp_fg[q, p, g]) / 2
    return trapz, grid, st
end

function get_gausslegendre(ebm, ps, st)
    """Get Gauss-Legendre nodes and weights for prior's domain"""
    
    a, b = minimum(st[Symbol("1")].grid; dims=2), maximum(st[Symbol("1")].grid; dims=2)
    
    no_grid = (ebm.fcns_qp[Symbol("1")].spline_string == "FFT" || 
        ebm.fcns_qp[Symbol("1")].spline_string == "Cheby" ||
        ebm.fcns_qp[Symbol("1")].spline_string == "Gottlieb"
    )
    
    if no_grid
        a = fill(half_quant(first(ebm.fcns_qp[Symbol("1")].grid_range)), size(a)) |> device
        b = fill(half_quant(last(ebm.fcns_qp[Symbol("1")].grid_range)), size(b)) |> device
    end
    
    nodes = (a + b) ./ 2 .+ (b - a) ./ 2 .* device(ebm.nodes)
    weights = (b - a) ./ 2 .* device(ebm.weights)
    nodes_cpu = cpu_device()(nodes)
    
    return nodes, weights, nodes_cpu
end

function gausslegendre_quadrature(
    ebm, 
    ps, 
    st; 
    ε::T=eps(half_quant),
    component_mask::Union{AbstractArray{T}, Nothing}=nothing
    ) where {T<:half_quant}
    """Gauss-Legendre quadrature for numerical integration"""

    nodes, weights, nodes_cpu = @ignore_derivatives get_gausslegendre(ebm, ps, st)
    π_nodes = ebm.prior_type == "learnable_gaussian" ? ebm.π_pdf(nodes', ps, ε) : ebm.π_pdf(nodes, ε)
    π_nodes = ebm.prior_type == "learnable_gaussian" ? π_nodes' : π_nodes

    # Energy function of each component
    nodes, st = prior_fwd(ebm, ps, st, nodes)

    # CDF evaluated by trapezium rule for integration; w_i * u(z_i)
    if component_mask !== nothing
        @tullio trapz[q, b, g] := (exp(nodes[q, p, g]) * π_nodes[q, g] * component_mask[q, p, b]) 
        @tullio trapz[q, b, g] = trapz[q, b, g] * weights[q, g] 
        return trapz, nodes_cpu, st
    else
        @tullio trapz[q, p, g] := (exp(nodes[q, p, g]) * π_nodes[p, g]) * weights[p, g]
        return trapz, nodes_cpu, st
    end
end

function init_ebm_prior(
    conf::ConfParse;
    prior_seed::Int=1,
    )
    widths = (
        try 
            parse.(Int, retrieve(conf, "EBM_PRIOR", "layer_widths"))
        catch
            parse.(Int, split(retrieve(conf, "EBM_PRIOR", "layer_widths"), ","))
        end
    )

    spline_degree = parse(Int, retrieve(conf, "EBM_PRIOR", "spline_degree"))
    layernorm = parse(Bool, retrieve(conf, "EBM_PRIOR", "layer_norm"))
    base_activation = retrieve(conf, "EBM_PRIOR", "base_activation")
    spline_function = retrieve(conf, "EBM_PRIOR", "spline_function")
    grid_size = parse(Int, retrieve(conf, "EBM_PRIOR", "grid_size"))
    grid_update_ratio = parse(half_quant, retrieve(conf, "EBM_PRIOR", "grid_update_ratio"))
    ε_scale = parse(half_quant, retrieve(conf, "EBM_PRIOR", "ε_scale"))
    μ_scale = parse(full_quant, retrieve(conf, "EBM_PRIOR", "μ_scale"))
    σ_base = parse(full_quant, retrieve(conf, "EBM_PRIOR", "σ_base"))
    σ_spline = parse(full_quant, retrieve(conf, "EBM_PRIOR", "σ_spline"))
    init_τ = parse(full_quant, retrieve(conf, "EBM_PRIOR", "init_τ"))
    τ_trainable = parse(Bool, retrieve(conf, "EBM_PRIOR", "τ_trainable"))
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size")) 
    τ_trainable = spline_function == "B-spline" ? false : τ_trainable
    reg = parse(full_quant, retrieve(conf, "EBM_PRIOR", "λ_reg"))

    P, Q = first(widths), last(widths)

    grid_range = parse.(half_quant, retrieve(conf, "EBM_PRIOR", "grid_range"))
    prior_type = retrieve(conf, "EBM_PRIOR", "π_0")
    mixture_model = parse(Bool, retrieve(conf, "EBM_PRIOR", "mixture_model"))
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

    quad_type = retrieve(conf, "EBM_PRIOR", "quadrature_method")
    quadrature_method = Dict(
        "gausslegendre" => (m, p, s, mask) -> gausslegendre_quadrature(m, p, s; ε=eps, component_mask=mask),
        "trapezium" => (m, p, s, mask) -> trapezium_quadrature(m, p, s; ε=eps, component_mask=mask),
    )[quad_type]

    N_quad = parse(Int, retrieve(conf, "EBM_PRIOR", "GaussQuad_nodes"))
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
        
    return ebm_prior(
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

function Lux.initialparameters(rng::AbstractRNG, prior::ebm_prior)
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
 
function Lux.initialstates(rng::AbstractRNG, prior::ebm_prior)
    st = NamedTuple(Symbol("$i") => Lux.initialstates(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)

    if prior.layernorm 
        for i in 1:prior.depth-1
            @reset st[Symbol("ln_$i")] = Lux.initialstates(rng, prior.fcns_qp[Symbol("ln_$i")]) |> hq
        end
    end

    return st 
end

end