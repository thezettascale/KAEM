module ebm_mix_prior

export mix_prior, init_mix_prior, log_prior

using CUDA, KernelAbstractions, Tullio, FastGaussQuadrature
using ConfParser, Random, Distributions, Lux, Accessors, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays
using NNlib: softmax, sigmoid_fast
using ChainRules: @ignore_derivatives

include("univariate_functions.jl")
include("inversion_sampling.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng, half_quant, full_quant, removeZero, removeNeg, hq, fq
using .InverseSampling: sample_prior, prior_fwd

prior_pdf = Dict(
    "uniform" => z -> half_quant.(0 .<= z .<= 1) |> device,
    "gaussian" => z -> half_quant(1 ./ sqrt(2π)) .* exp.(-z.^2 ./ 2),
    "lognormal" => (z, ε) -> exp.(-(log.(z .+ ε)).^2 ./ 2) ./ (z .* half_quant(sqrt(2π)) .+ ε),
)

struct mix_prior <: Lux.AbstractLuxLayer
    fcns_qp::NamedTuple
    layernorm::Bool
    depth::Int
    prior_type::AbstractString
    π_pdf::Function
    sample_z::Function
    contrastive_div::Bool
    λ::half_quant
    q_size::Int
    p_size::Int
    quadrature_method::AbstractString
    N_quad::Int
    nodes::AbstractArray{half_quant}
    weights::AbstractArray{full_quant}
end

function trapezium_quadrature(
    mix,
    ps,
    st;
    ε::full_quant=eps(full_quant)
)
    """
    Approximate the partition function of the mixture ebm-prior using trapezium rule.

    ∫ exp(f(z)) π_0(z) dz ≈ ∑_g 0.5(Δg) [exp(f(z_{q,g})) π_0(z_{q,g}) + exp(f(z_{q,g+1})) π_0(z_{q,g+1})]

    Args:
        mix: The mixture ebm-prior.
        ps: The parameters of the mixture ebm-prior.
        st: The states of the mixture ebm-prior.

    Returns:
        The log-partition function of the mixture ebm-prior.
        The updated state
    """
    grid = st[Symbol("1")].grid
    grid_size = size(grid,2)
    π_g = mix.prior_type == "lognormal" ? mix.π_pdf(grid, ε) : mix.π_pdf(grid)
    log_π_grid, Δg = log.(π_g .+ ε), grid[:, 2:end] - grid[:, 1:end-1] 

    # Energy function of each component, q -> p
    grid, st = prior_fwd(mix, ps, st, grid)
    @tullio trapz[q,g,p] := grid[q,p,g] + log_π_grid[q,g]
    trapz = Δg .* (trapz[:, 2:end, :] + trapz[:, 1:end-1, :]) ./ 2
    return dropdims(sum(trapz, dims=2); dims=2), st
end

function gauss_quadrature(
    mix,
    ps,
    st;
    ε::full_quant=eps(full_quant)
    )
    """
    Approximate the partition function of the mixture ebm-prior using gauss-quadrature.
    
     ∫ exp(f(z)) π_0(z) dz ≈ ∑_i w_i exp(f(z_i)) π_0(z_i)

     Args:
        mix: The mixture ebm-prior.
        ps: The parameters of the mixture ebm-prior.
        st: The states of the mixture ebm-prior.

    Returns:
        The log-partition function of the mixture ebm-prior.
        The updated state
    """
    # Map domains
    a, b = mix.fcns_qp[Symbol("1")].grid_range
    nodes = (a + b) ./ 2 .+ (b - a) ./ 2 .* mix.nodes |> device
    weights = (b - a) ./ 2 .* mix.weights |> device
    π_nodes = mix.prior_type == "lognormal" ? mix.π_pdf(nodes, ε) : mix.π_pdf(nodes)
    log_π_nodes = log.(π_nodes .+ ε)

    # Energy function of each component, q -> p
    nodes, st = prior_fwd(mix, ps, st, nodes)
    @tullio trapz[q,p] := (nodes[q,p,g] + log_π_nodes[q,g]) * weights[1, g]
    return trapz, st
end

log_partition_function = (mix, ps, st; ε=eps(full_quant)) -> mix.quadrature_method == "trapezium" ? trapezium_quadrature(mix, ps, st; ε=ε) : gauss_quadrature(mix, ps, st; ε=ε)

function log_prior(
    mix, 
    z::AbstractArray{half_quant},
    ps, 
    st;
    normalize::Bool=false,
    ε::full_quant=eps(full_quant),
    agg::Bool=true
    )
    """
    Evaluate the unnormalized log-probability of the mixture ebm-prior.
    The likelihood of samples from each mixture model, z_q, is evaluated 
    for all components of the mixture model it has been sampled from , M_q.

    ∑_q [ log ( ∑_p α_p exp(f_{q,p}(z_q)) π_0(z_q) ) ]
    
    Args:
        mix: The mixture ebm-prior.
        z: The component-wise latent samples to evaulate the measure on, (num_samples, q)
        ps: The parameters of the mixture ebm-prior.
        st: The states of the mixture ebm-prior.
        normalize: Whether to normalize the log-probability.
        ε: The small value to avoid log(0).
        agg: Whether to sum the log-probability over the samples.

    Returns:
        The unnormalized log-probability of the mixture ebm-prior.
        The L1 regularization term.
        The updated states of the mixture ebm-prior.
    """
    b_size = size(z,2)
    
    # Mixture proportions and prior
    alpha = softmax(ps[Symbol("α")]; dims=2) 
    π_0 = mix.prior_type == "lognormal" ? mix.π_pdf(z, ε) : mix.π_pdf(z)
    @tullio log_απ[q,p,b] := log(alpha[q,p] * π_0[q,b] + ε)

    # Energy functions of each component, q -> p
    for i in 1:mix.depth
        z = fwd(mix.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = i == 1 ? reshape(z, size(z,2), b_size*mix.q_size) : dropdims(sum(z, dims=1); dims=1)

        if mix.layernorm && i < mix.depth
            z, st_new = Lux.apply(mix.fcns_qp[Symbol("ln_$i")], z, ps[Symbol("ln_$i")], st[Symbol("ln_$i")])
            @reset st[Symbol("ln_$i")] = st_new
        end
    end
    z = reshape(z, mix.q_size, mix.p_size, b_size)

    # Unnormalized or normalized log-probability
    logprob = z + log_απ
    norm, st = normalize ? log_partition_function(mix, ps, st; ε=ε) : (half_quant(0), st)
    logprob = logprob .- norm
    l1_reg = mix.λ * sum(abs.(ps[Symbol("α")])) |> fq # L1 regularization to encourage sparsity
    return dropdims(sum(logprob |> fq; dims=(1,2)); dims=(1,2)) .+ l1_reg, st
end

function init_mix_prior(
    conf::ConfParse;
    prior_seed::Int=1,
    )
    widths = (
        try 
            parse.(Int, retrieve(conf, "MIX_PRIOR", "layer_widths"))
        catch
            parse.(Int, split(retrieve(conf, "MIX_PRIOR", "layer_widths"), ","))
        end
    )

    widths = reverse(widths)
    spline_degree = parse(Int, retrieve(conf, "MIX_PRIOR", "spline_degree"))
    layernorm = parse(Bool, retrieve(conf, "MIX_PRIOR", "layer_norm"))
    base_activation = retrieve(conf, "MIX_PRIOR", "base_activation")
    spline_function = retrieve(conf, "MIX_PRIOR", "spline_function")
    grid_size = parse(Int, retrieve(conf, "MIX_PRIOR", "grid_size"))
    grid_update_ratio = parse(half_quant, retrieve(conf, "MIX_PRIOR", "grid_update_ratio"))
    grid_range = parse.(half_quant, retrieve(conf, "MIX_PRIOR", "grid_range"))
    ε_scale = parse(half_quant, retrieve(conf, "MIX_PRIOR", "ε_scale"))
    μ_scale = parse(full_quant, retrieve(conf, "MIX_PRIOR", "μ_scale"))
    σ_base = parse(full_quant, retrieve(conf, "MIX_PRIOR", "σ_base"))
    σ_spline = parse(full_quant, retrieve(conf, "MIX_PRIOR", "σ_spline"))
    init_τ = parse(full_quant, retrieve(conf, "MIX_PRIOR", "init_τ"))
    τ_trainable = parse(Bool, retrieve(conf, "MIX_PRIOR", "τ_trainable"))
    τ_trainable = spline_function == "B-spline" ? false : τ_trainable
    prior_type = retrieve(conf, "MIX_PRIOR", "π_0")
    contrastive_divergence = parse(Bool, retrieve(conf, "TRAINING", "contrastive_divergence_training"))
    eps = parse(half_quant, retrieve(conf, "TRAINING", "eps"))
    λ = parse(half_quant, retrieve(conf, "MIX_PRIOR", "l1_regularization"))
    
    sample_function = (m, n, p, s, seed) -> @ignore_derivatives sample_prior(m, n, p, Lux.testmode(s); seed=seed, ε=eps)
    
    functions = NamedTuple()
    for i in eachindex(widths[1:end-1])
        prior_seed, rng = next_rng(prior_seed)
        base_scale = (μ_scale * (full_quant(1) / √(full_quant(widths[i])))
        .+ σ_base .* (randn(rng, full_quant, widths[i], widths[i+1]) .* full_quant(2) .- full_quant(1)) .* (full_quant(1) / √(full_quant(widths[i]))))

        func = init_function(
        widths[i],
        widths[i+1];
        spline_degree=spline_degree,
        base_activation=base_activation,
        spline_function=spline_function,
        grid_size=grid_size,
        grid_update_ratio=grid_update_ratio,
        grid_range=Tuple(grid_range),
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

    quadrature_method = retrieve(conf, "MIX_PRIOR", "quadrature_method")
    N_quad = parse(Int, retrieve(conf, "MIX_PRIOR", "GaussQuad_nodes"))
    nodes, weights = gausslegendre(N_quad)
    nodes = repeat(nodes', first(widths), 1) .|> half_quant
    weights = full_quant.(weights)'

    return mix_prior(functions, layernorm, length(widths)-1, prior_type, prior_pdf[prior_type], sample_function, contrastive_divergence, λ, first(widths), last(widths), quadrature_method, N_quad, nodes, weights)
end

function Lux.initialparameters(rng::AbstractRNG, prior::mix_prior)
    ps = NamedTuple(Symbol("$i") => Lux.initialparameters(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)
    @reset ps[Symbol("α")] = glorot_uniform(rng, full_quant, prior.q_size, prior.p_size)

    if prior.layernorm 
        for i in 1:prior.depth-1
            @reset ps[Symbol("ln_$i")] = Lux.initialparameters(rng, prior.fcns_qp[Symbol("ln_$i")]) 
        end
    end

    return ps 
end
 
function Lux.initialstates(rng::AbstractRNG, prior::mix_prior)
    st = NamedTuple(Symbol("$i") => Lux.initialstates(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)

    if prior.layernorm 
        for i in 1:prior.depth-1
            @reset st[Symbol("ln_$i")] = Lux.initialstates(rng, prior.fcns_qp[Symbol("ln_$i")]) |> hq
        end
    end

    return st 
end

end