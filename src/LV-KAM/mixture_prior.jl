module ebm_mix_prior

export mix_prior, init_mix_prior, log_prior

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Distributions, Lux, Accessors, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays
using NNlib: softmax, sigmoid_fast
using ChainRules: @ignore_derivatives
using LogExpFunctions: logsumexp

include("univariate_functions.jl")
include("inversion_sampling.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng, quant
using .InverseSampling: sample_prior

prior_distributions = Dict(
    "uniform" => Uniform(0f0,1f0),
    "gaussian" => Normal(0f0,1f0),
)

prior_pdf = Dict(
    "uniform" => z -> quant.(0 .<= z .<= 1) |> device,
    "gaussian" => z -> 1 ./ sqrt(2π) .* exp.(-z.^2 ./ 2),
)

struct mix_prior <: Lux.AbstractLuxLayer
    fcns_qp::NamedTuple
    depth::Int
    π_0::Union{Uniform, Normal, Bernoulli}
    π_pdf::Function
    sample_z::Function
    contrastive_div::Bool
end

function log_partition_function(
    mix::mix_prior,
    ps,
    st;
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
    """
    grid = mix.fcns_qp[Symbol("1")].grid'
    grid_size, q_size = size(grid)
    π_grid, Δg = mix.π_pdf(grid), grid[2:end, :] - grid[1:end-1, :] 
    
    for i in 1:mix.depth
        grid = fwd(mix.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], grid)
        grid = i == 1 ? reshape(grid, grid_size*q_size, size(grid, 3)) : dropdims(sum(grid, dims=2); dims=2)
    end
    grid = reshape(grid, grid_size, q_size, size(grid, 2))
    grid = exp.(grid) .* π_grid
    trapz = 5f-1 .* Δg .* (grid[2:end, :, :] + grid[1:end-1, :, :])
    return log.(sum(trapz, dims=1) .+ eps(eltype(trapz)))
end

function log_prior(
    mix::mix_prior, 
    z::AbstractArray, 
    ps, 
    st;
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

    Returns:
        The unnormalized log-probability of the mixture ebm-prior.
    """
    b_size, q_size, p_size = size(z)..., mix.fcns_qp[Symbol("$(mix.depth)")].out_dim
    
    # Mixture proportions and prior
    alpha = softmax(ps[Symbol("α")]; dims=2) 
    π_0 = mix.π_pdf(z)
    @tullio log_απ[b,q,p] := alpha[q,p] * π_0[b,q]
    log_απ = log.(log_απ .+ eps(eltype(log_απ)))

    # Energy functions of each component, q -> p
    for i in 1:mix.depth
        z = fwd(mix.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = i == 1 ? reshape(z, b_size*q_size, size(z, 3)) : dropdims(sum(z, dims=2); dims=2)
    end
    z = reshape(z, b_size, q_size, p_size)

    # Unnormalized log-probability with logsumexp trick
    z = log_απ .+ z 
    z = !mix.contrastive_div ? z .- log_partition_function(mix, ps, st) : z
    M = maximum(z, dims=3)
    z = sum(M .+ logsumexp(z .- M, dims=3); dims=2)
    return dropdims(z; dims=(2,3))
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
    base_activation = retrieve(conf, "MIX_PRIOR", "base_activation")
    spline_function = retrieve(conf, "MIX_PRIOR", "spline_function")
    grid_size = parse(Int, retrieve(conf, "MIX_PRIOR", "grid_size"))
    grid_update_ratio = parse(quant, retrieve(conf, "MIX_PRIOR", "grid_update_ratio"))
    grid_range = parse.(quant, retrieve(conf, "MIX_PRIOR", "grid_range"))
    ε_scale = parse(quant, retrieve(conf, "MIX_PRIOR", "ε_scale"))
    μ_scale = parse(quant, retrieve(conf, "MIX_PRIOR", "μ_scale"))
    σ_base = parse(quant, retrieve(conf, "MIX_PRIOR", "σ_base"))
    σ_spline = parse(quant, retrieve(conf, "MIX_PRIOR", "σ_spline"))
    init_η = parse(quant, retrieve(conf, "MIX_PRIOR", "init_η"))
    η_trainable = parse(Bool, retrieve(conf, "MIX_PRIOR", "η_trainable"))
    η_trainable = spline_function == "B-spline" ? false : η_trainable
    prior_type = retrieve(conf, "MIX_PRIOR", "π_0")
    contrastive_divergence = parse(Bool, retrieve(conf, "TRAINING", "contrastive_divergence_training"))
    
    sample_function = (m, n, p, s, seed) -> @ignore_derivatives sample_prior(m, n, p, s; seed=seed)
    
    functions = NamedTuple()
    for i in eachindex(widths[1:end-1])
        prior_seed, rng = next_rng(prior_seed)
        base_scale = (μ_scale * (1f0 / √(quant(widths[i])))
        .+ σ_base .* (randn(rng, quant, widths[i], widths[i+1]) .* 2f0 .- 1f0) .* (1f0 / √(quant(widths[i]))))

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
        init_η=init_η,
        η_trainable=η_trainable,
        )

        @reset functions[Symbol("$i")] = func
    end

    return mix_prior(functions, length(widths)-1, prior_distributions[prior_type], prior_pdf[prior_type], sample_function, contrastive_divergence)
end

function Lux.initialparameters(rng::AbstractRNG, prior::mix_prior)
    q_size = prior.fcns_qp[Symbol("1")].in_dim
    p_size = prior.fcns_qp[Symbol("$(prior.depth)")].out_dim
    ps = NamedTuple(Symbol("$i") => Lux.initialparameters(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)
    @reset ps[Symbol("α")] = glorot_normal(rng, quant, q_size, p_size)
    return ps
end
 
function Lux.initialstates(rng::AbstractRNG, prior::mix_prior)
    st = NamedTuple(Symbol("$i") => Lux.initialstates(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)
    return st
end

end