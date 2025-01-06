module ebm_mix_prior

export mix_prior, init_mix_prior, log_prior

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Distributions, Lux, Accessors, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays
using NNlib: softmax, sigmoid_fast, logsumexp
using ChainRules: @ignore_derivatives

include("univariate_functions.jl")
include("inversion_sampling.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng
using .InverseSampling: sample_prior

prior_distributions = Dict(
    "uniform" => Uniform(0f0,1f0),
    "gaussian" => Normal(0f0,1f0),
)

prior_pdf = Dict(
    "uniform" => z -> Float32.(0 .<= z .<= 1) |> device,
    "gaussian" => z -> 1 ./ sqrt(2π) .* exp.(-z.^2 ./ 2),
)

struct mix_prior <: Lux.AbstractLuxLayer
    fcns_qp::NamedTuple
    depth::Int
    π_0::Union{Uniform, Normal, Bernoulli}
    π_pdf::Function
    sample_z::Function
end

function log_prior(
    mix::mix_prior, 
    z::AbstractArray, 
    ps, 
    st;
    )
    """
    Compute the unnormalized log-probability of the mixture ebm-prior.
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
    
    # Mixture proportions and prior, prepared for logsumexp trick
    alpha = softmax(ps[Symbol("α")]; dims=2) 
    π_0 = mix.π_pdf(z)
    log_α_prior = log.(@tullio(alphaprior[b, q, p] := alpha[q, p] * π_0[b, q]) .+ eps(eltype(z)))

    # Energy functions of each component, q -> p
    for i in 1:mix.depth
        z = fwd(mix.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = i == 1 ? reshape(z, b_size*q_size, size(z, 3)) : sum(z, dims=2)[:, 1, :]
    end
    z = reshape(z, b_size, q_size, p_size)

    # Using logsumexp trick to avoid underflow/overflow
    z = z + log_α_prior
    max_z = maximum(z; dims=3)
    z = logsumexp(z .- max_z; dims=3) .+ max_z
    return z[:,:]
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
    grid_update_ratio = parse(Float32, retrieve(conf, "MIX_PRIOR", "grid_update_ratio"))
    grid_range = parse.(Float32, retrieve(conf, "MIX_PRIOR", "grid_range"))
    ε_scale = parse(Float32, retrieve(conf, "MIX_PRIOR", "ε_scale"))
    μ_scale = parse(Float32, retrieve(conf, "MIX_PRIOR", "μ_scale"))
    σ_base = parse(Float32, retrieve(conf, "MIX_PRIOR", "σ_base"))
    σ_spline = parse(Float32, retrieve(conf, "MIX_PRIOR", "σ_spline"))
    init_η = parse(Float32, retrieve(conf, "MIX_PRIOR", "init_η"))
    η_trainable = parse(Bool, retrieve(conf, "MIX_PRIOR", "η_trainable"))
    η_trainable = spline_function == "B-spline" ? false : η_trainable
    prior_type = retrieve(conf, "MIX_PRIOR", "π_0")
    interpolate = parse(Bool, retrieve(conf, "MIX_PRIOR", "inversion_sampling_interpolate"))
    
    sample_function = (m, n, p, s, seed) -> @ignore_derivatives sample_prior(m, n, p, s; seed=seed, interpolation=interpolate)
    
    functions = NamedTuple()
    for i in eachindex(widths[1:end-1])
        prior_seed, rng = next_rng(prior_seed)
        base_scale = (μ_scale * (1f0 / √(Float32(widths[i])))
        .+ σ_base .* (randn(rng, Float32, widths[i], widths[i+1]) .* 2f0 .- 1f0) .* (1f0 / √(Float32(widths[i]))))

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

    return mix_prior(functions, length(widths)-1, prior_distributions[prior_type], prior_pdf[prior_type], sample_function)
end

function Lux.initialparameters(rng::AbstractRNG, prior::mix_prior)
    q_size = prior.fcns_qp[Symbol("1")].in_dim
    p_size = prior.fcns_qp[Symbol("$(prior.depth)")].out_dim
    ps = NamedTuple(Symbol("$i") => Lux.initialparameters(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)
    @reset ps[Symbol("α")] = zeros(Float32, q_size, p_size)
    return ps
end
 
function Lux.initialstates(rng::AbstractRNG, prior::mix_prior)
    st = NamedTuple(Symbol("$i") => Lux.initialstates(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)
    return st
end

end