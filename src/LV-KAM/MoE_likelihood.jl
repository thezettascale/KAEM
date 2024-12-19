module MoE_likelihood

export MoE_lkhood, init_MoE_lkhood, log_likelihood, generate_from_z, importance_sampler

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Distributions, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays, Accessors
using NNlib: softmax, sigmoid_fast, tanh_fast
using ChainRules: @ignore_derivatives

include("univariate_functions.jl")
include("mixture_prior.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng
using .ebm_mix_prior

activation_mapping = Dict(
    "tanh" => tanh_fast,
    "sigmoid" => sigmoid_fast,
    "none" => x -> x 
)

lkhood_models = Dict(
    "l2" => (x, x̂) -> - @tullio(l2[b, s] := -(x[b, i] - x̂[s, i])^2),
    "bernoulli" => (x, x̂) -> @tullio(bern[b, s] := x[b, i] * log(x̂[s, i] + 1f-4) + (1 - x[b, i]) * log(1 - x̂[s, i] + 1f-4)),
)

struct MoE_lkhood <: Lux.AbstractLuxLayer
    Λ_fcns::NamedTuple
    depth::Int
    out_size::Int
    σ_ε::Float32
    σ_llhood::Float32
    log_lkhood_model::Function
    output_activation::Function
    weight_fcn::Function
    resamples::Int
    ess_thresh::Float32
end

function init_MoE_lkhood(
    conf::ConfParse,
    output_dim::Int;
    lkhood_seed::Int=1,
    weight_fcn::Function= x -> softmax(x; dims=2) 
    )

    widths = parse.(Int, retrieve(conf, "MOE_LIKELIHOOD", "layer_widths"))
    widths = (widths..., 1)
    first(widths) !== last(parse.(Int, retrieve(conf, "MIX_PRIOR", "layer_widths"))) && (
        error("First width must be equal to the hidden dimension of the prior.")
    )

    spline_degree = parse(Int, retrieve(conf, "MOE_LIKELIHOOD", "spline_degree"))
    base_activation = retrieve(conf, "MOE_LIKELIHOOD", "base_activation")
    spline_function = retrieve(conf, "MOE_LIKELIHOOD", "spline_function")
    grid_size = parse(Int, retrieve(conf, "MOE_LIKELIHOOD", "grid_size"))
    grid_update_ratio = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "grid_update_ratio"))
    grid_range = parse.(Float32, retrieve(conf, "MOE_LIKELIHOOD", "grid_range"))
    ε_scale = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "ε_scale"))
    μ_scale = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "μ_scale"))
    σ_base = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "σ_base"))
    σ_spline = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "σ_spline"))
    init_η = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "init_η"))
    η_trainable = parse(Bool, retrieve(conf, "MOE_LIKELIHOOD", "η_trainable"))
    η_trainable = spline_function == "B-spline" ? false : η_trainable
    noise_var = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "generator_noise_variance"))
    gen_var = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "generator_variance"))
    lkhood_model = retrieve(conf, "MOE_LIKELIHOOD", "likelihood_model")
    output_act = retrieve(conf, "MOE_LIKELIHOOD", "output_activation")
    resample_size = parse(Int, retrieve(conf, "MOE_LIKELIHOOD", "importance_resample_size"))
    resampling_ess_threshold = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "resampling_ess_threshold"))

    functions = NamedTuple()
    for i in eachindex(widths[1:end-1])
        lkhood_seed = next_rng(lkhood_seed)
        base_scale = (μ_scale * (1f0 / √(Float32(widths[i])))
        .+ σ_base .* (randn(Float32, widths[i], widths[i+1]) .* 2f0 .- 1f0) .* (1f0 / √(Float32(widths[i]))))

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
        
    return MoE_lkhood(functions, length(widths)-1, output_dim, noise_var, gen_var, lkhood_models[lkhood_model], activation_mapping[output_act], weight_fcn, resample_size, resampling_ess_threshold)
end

function Lux.initialparameters(rng::AbstractRNG, lkhood::MoE_lkhood)
    ps = NamedTuple(Symbol("$i") => Lux.initialparameters(rng, lkhood.Λ_fcns[Symbol("$i")]) for i in 1:lkhood.depth)
    @reset ps[Symbol("w")] = glorot_normal(rng, Float32, lkhood.Λ_fcns[Symbol("1")].in_dim, lkhood.out_size)
    return ps
end

function Lux.initialstates(rng::AbstractRNG, lkhood::MoE_lkhood)
    st = NamedTuple(Symbol("$i") => Lux.initialstates(rng, lkhood.Λ_fcns[Symbol("$i")]) for i in 1:lkhood.depth)
    return st
end

function generate_from_z(
    lkhood::MoE_lkhood, 
    ps, 
    st, 
    z::AbstractArray; 
    seed::Int=1
    )
    """
    Generate data from the likelihood model.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        seed: The seed for the random number generator.

    Returns:
        The generated data.
        The updated seed.
    """
    # Gating function, feature-specific
    gate_w = ps[Symbol("w")]
    wz = @tullio out[b, q, o] := z[b, q] * gate_w[q, o]
    γ = softmax(wz; dims=2)

    # Gen function, experts for all features
    for i in 1:lkhood.depth
        z = fwd(lkhood.Λ_fcns[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = i == 1 ? @views(reshape(z, prod(size(wz)[1:2]), size(z, 3))) : sum(z, dims=2)[:, 1, :] 
    end
    z = @views(reshape(z, size(wz)[1:2]..., 1))

    seed = next_rng(seed)
    ε = rand(Normal(0f0, lkhood.σ_ε), size(lkhood.out_size)) |> device

    # Generate data
    x̂ = @tullio gen[b, q, o] := z[b, q, 1] * γ[b, q, o]
    x̂ = sum(x̂, dims=2)[:, 1, :] .+ ε
    return lkhood.output_activation(x̂), seed
end

function log_likelihood(
    lkhood::MoE_lkhood, 
    ps, 
    st, 
    x::AbstractArray, 
    z::AbstractArray;
    seed::Int=1
    )
    """
    Compute the log-likelihood of the data given the latent variable.
    The updated seed is not returned, since noise is ignored by derivatives anyway.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data, (batch_size, out_dim).
        z: The latent variable, (batch_size, q).
        seed: The seed for the random number generator.

    Returns:
        The log-likelihood of the batch given the latent samples.
    """
    
    x̂, seed = generate_from_z(lkhood, ps, st, z; seed=seed)
    return lkhood.log_lkhood_model(x, x̂) ./ (2f0*lkhood.σ_llhood^2)
end

function importance_sampler(
    lkhood::MoE_lkhood, 
    ps, 
    st, 
    x::AbstractArray, 
    z::AbstractArray;
    seed::Int=1
    )
    """
    Resample the latent variable using importance sampling weights.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data, (batch_size, out_dim).
        z: The latent variable, (batch_size, q).

    Returns:
        The resampled latent variable.
        The new log-likelihood.
        The importance sampling weights.
    """
    # Initial importance sampling weights
    logllhood = log_likelihood(lkhood, ps, st, x, z)
    init_weights = cpu_device()(softmax(sum(logllhood, dims=1)[1, :]))
    
    # Systematic resampling 
    ESS = 1 / sum(init_weights.^2)
    N = size(z, 1)
    if ESS < lkhood.ess_thresh*N
        
        cdf = cumsum(init_weights)
        
        seed = next_rng(seed)
        u0 = rand() / N
        u = u0 .+ (0:N-1) ./ N
        indices = zeros(Int, 0) |> device

        for i in 1:N
            indices = vcat(indices, findfirst(cdf .>= u[i]))
        end

        z = z[indices, :]
    end

    logllhood = log_likelihood(lkhood, ps, st, x, z)
    weights = softmax(logllhood; dims=2)

    return z, logllhood, weights
end

end