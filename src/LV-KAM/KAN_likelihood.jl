module KAN_likelihood

export KAN_lkhood, init_KAN_lkhood, log_likelihood, generate_from_z

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays, Accessors
using NNlib
using ChainRules: @ignore_derivatives

include("univariate_functions.jl")
include("mixture_prior.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng
using .ebm_mix_prior

output_activation_mapping = Dict(
    "tanh" => NNlib.tanh_fast,
    "sigmoid" => NNlib.sigmoid_fast,
    "none" => identity
)

gating_activation_mapping = Dict(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.tanh_fast,
    "sigmoid" => NNlib.sigmoid_fast,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu,
    "selu" => NNlib.selu,
    "tanh" => NNlib.tanh_fast,
    "silu" => x -> x .* NNlib.sigmoid_fast(x),
    "elu" => NNlib.elu,
    "celu" => NNlib.celu,
    "none" => identity
)

lkhood_models = Dict(
    "l2" => (x::AbstractArray, x̂::AbstractArray) -> -sum((x .- x̂).^2, dims=3)[:,:,1],
    "bernoulli" => (x::AbstractArray, x̂::AbstractArray) -> sum(x .* log.(x̂ .+ eps(eltype(x))) .+ (1 .- x) .* log.(1 .- x̂ .+ eps(eltype(x))), dims=3)[:,:,1],
)

struct KAN_lkhood <: Lux.AbstractLuxLayer
    Φ_fcns::NamedTuple
    depth::Int
    out_size::Int
    σ_ε::Float32
    σ_llhood::Float32
    log_lkhood_model::Function
    output_activation::Function
    gating_activation::Function
    resample_z::Function
end

function generate_from_z(
    lkhood::KAN_lkhood, 
    ps, 
    st, 
    z::AbstractArray;
    seed::Int=1,
    noise::Bool=true,
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
    num_samples, q_size = size(z)

    # KAN functions
    for i in 1:lkhood.depth
        z = fwd(lkhood.Φ_fcns[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = dropdims(sum(z, dims=2); dims=2)
    end
    
    # Add noise
    seed, rng = next_rng(seed)
    ε = lkhood.σ_ε * randn(rng, Float32, size(z)) |> device
    !noise && (ε .*= 0f0)
    return lkhood.output_activation(z + ε), seed
end

function log_likelihood(
    lkhood::KAN_lkhood, 
    ps, 
    st, 
    x::AbstractArray, 
    z::AbstractArray;
    seed::Int=1,
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
    logllhood = lkhood.log_lkhood_model(
        permutedims(x[:,:,:], [2,3,1]),
        permutedims(x̂[:,:,:], [3,1,2]),
    )
    return logllhood ./ (2f0*lkhood.σ_llhood^2), seed
end

function stratified_sampler(
    weights::AbstractArray;
    seed::Int=1
)
    """
    Resample the latent variable using stratified sampling.
    Args:
        weights: A matrix of weights where each row corresponds to a sample's weights.
        seed: Random seed for reproducibility.
    Returns:
        The resampled indices and the updated seed.
    """
    B, N = size(weights)
    cdf = permutedims(cumsum(weights[:,:,:], dims=2), [1,3,2])
        
    # Generate stratified thresholds
    seed, rng = next_rng(seed)
    u = (rand(B, N) .+ (0:N-1)') ./ N
    
    # Find resampled indices in a vectorised manner
    indices = sum(u[:, :, :] .> cdf, dims=3) .+ 1
    return dropdims(indices, dims=3), seed
end


function init_KAN_lkhood(
    conf::ConfParse,
    output_dim::Int;
    lkhood_seed::Int=1,
    )
    q_size = (
        try 
            last(parse.(Int, retrieve(conf, "MIX_PRIOR", "layer_widths")))
        catch
            last(parse.(Int, split(retrieve(conf, "MIX_PRIOR", "layer_widths"), ",")))
        end
    )

    expert_widths = (
        try 
            parse.(Int, retrieve(conf, "KAN_LIKELIHOOD", "expert_widths"))
        catch
            parse.(Int, split(retrieve(conf, "KAN_LIKELIHOOD", "expert_widths"), ","))
        end
    )

    expert_widths = (expert_widths..., output_dim)
    first(expert_widths) !== q_size && (error("First expert Φ_hidden_widths must be equal to the hidden dimension of the prior."))

    spline_degree = parse(Int, retrieve(conf, "KAN_LIKELIHOOD", "spline_degree"))
    base_activation = retrieve(conf, "KAN_LIKELIHOOD", "base_activation")
    spline_function = retrieve(conf, "KAN_LIKELIHOOD", "spline_function")
    grid_size = parse(Int, retrieve(conf, "KAN_LIKELIHOOD", "grid_size"))
    grid_update_ratio = parse(Float32, retrieve(conf, "KAN_LIKELIHOOD", "grid_update_ratio"))
    grid_range = parse.(Float32, retrieve(conf, "KAN_LIKELIHOOD", "grid_range"))
    ε_scale = parse(Float32, retrieve(conf, "KAN_LIKELIHOOD", "ε_scale"))
    μ_scale = parse(Float32, retrieve(conf, "KAN_LIKELIHOOD", "μ_scale"))
    σ_base = parse(Float32, retrieve(conf, "KAN_LIKELIHOOD", "σ_base"))
    σ_spline = parse(Float32, retrieve(conf, "KAN_LIKELIHOOD", "σ_spline"))
    init_η = parse(Float32, retrieve(conf, "KAN_LIKELIHOOD", "init_η"))
    η_trainable = parse(Bool, retrieve(conf, "KAN_LIKELIHOOD", "η_trainable"))
    η_trainable = spline_function == "B-spline" ? false : η_trainable
    noise_var = parse(Float32, retrieve(conf, "KAN_LIKELIHOOD", "generator_noise_var"))
    gen_var = parse(Float32, retrieve(conf, "KAN_LIKELIHOOD", "generator_variance"))
    lkhood_model = retrieve(conf, "KAN_LIKELIHOOD", "likelihood_model")
    output_act = retrieve(conf, "KAN_LIKELIHOOD", "output_activation")
    gating_act = retrieve(conf, "KAN_LIKELIHOOD", "gating_activation")

    resample_function = (weights, seed) -> @ignore_derivatives stratified_sampler(weights; seed=seed)

    initialize_function = (in_dim, out_dim, base_scale) -> init_function(
        in_dim,
        out_dim;
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

    # KAN functions
    Φ_functions = NamedTuple() # Expert functions
    for i in eachindex(expert_widths[1:end-1])
        lkhood_seed, rng = next_rng(lkhood_seed)
        base_scale = (μ_scale * (1f0 / √(Float32(expert_widths[i])))
        .+ σ_base .* (randn(rng, Float32, expert_widths[i], expert_widths[i+1]) .* 2f0 .- 1f0) .* (1f0 / √(Float32(expert_widths[i]))))
        @reset Φ_functions[Symbol("$i")] = initialize_function(expert_widths[i], expert_widths[i+1], base_scale)
    end

    return KAN_lkhood(Φ_functions, length(expert_widths)-1, output_dim, noise_var, gen_var, lkhood_models[lkhood_model], output_activation_mapping[output_act], gating_activation_mapping[gating_act], resample_function)
end

function Lux.initialparameters(rng::AbstractRNG, lkhood::KAN_lkhood)
    ps = NamedTuple(Symbol("$i") => Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("$i")]) for i in 1:lkhood.depth)
    return ps
end

function Lux.initialstates(rng::AbstractRNG, lkhood::KAN_lkhood)
    st = NamedTuple(Symbol("$i") => Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("$i")]) for i in 1:lkhood.depth)
    return st
end

end