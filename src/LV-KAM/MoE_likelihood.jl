module MoE_likelihood

export MoE_lkhood, init_MoE_lkhood, log_likelihood, generate_from_z, importance_sampler

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays, Accessors
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
    "none" => identity
)

lkhood_models = Dict(
    "l2" => (x::AbstractArray, x̂::AbstractArray) -> -sum((x .- x̂).^2, dims=3)[:,:,1],
    "bernoulli" => (x::AbstractArray, x̂::AbstractArray) -> sum(x .* log.(x̂ .+ eps(eltype(x))) .+ (1 .- x) .* log.(1 .- x̂ .+ eps(eltype(x))), dims=3)[:,:,1],
)

struct MoE_lkhood <: Lux.AbstractLuxLayer
    Ω_fcns::NamedTuple
    Λ_fcns::NamedTuple
    depth::Int
    out_size::Int
    σ_ε::Float32
    σ_llhood::Float32
    log_lkhood_model::Function
    output_activation::Function
    resample_z::Function
end

function generate_from_z(
    lkhood::MoE_lkhood, 
    ps, 
    st, 
    z::AbstractArray
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
    w = ps[Symbol("gate_w")] 
    w = z * w
    w = permutedims(w[:,:,:], [1,3,2])

    # Ω gating function, and Λ function; feature-specific; samples x q x o
    Ω, Λ = z, z
    for i in 1:lkhood.depth
        Ω = fwd(lkhood.Ω_fcns[Symbol("Ω_$i")], ps[Symbol("Ω_$i")], st[Symbol("Ω_$i")], Ω)
        Ω = i == 1 ? reshape(Ω, num_samples*q_size, size(Ω, 3)) : sum(Ω, dims=2)[:, 1, :] 

        Λ = fwd(lkhood.Λ_fcns[Symbol("Λ_$i")], ps[Symbol("Λ_$i")], st[Symbol("Λ_$i")], Λ)
        Λ = i == 1 ? reshape(Λ, num_samples*q_size, size(Λ, 3)) : sum(Λ, dims=2)[:, 1, :] 
    end
    Ω, Λ = reshape(Ω, num_samples, q_size, 1), reshape(Λ, num_samples, q_size, 1)
    
    # Attention-like gating
    γ = softmax(w .* Ω ./ Float32(sqrt(q_size)); dims=2)

    # Generate data
    @tullio x̂[b,o] := (Λ[b,q,1] * γ[b,q,o]) 
    return lkhood.output_activation(x̂)
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
    x̂ = generate_from_z(lkhood, ps, st, z)
    logllhood = lkhood.log_lkhood_model(
        permutedims(x[:,:,:], [2,3,1]),
        permutedims(x̂[:,:,:], [3,1,2])
    )

    seed, rng = next_rng(seed)
    ε = lkhood.σ_ε * randn(rng, Float32, size(logllhood)) |> device

    return (logllhood ./ (2f0*lkhood.σ_llhood^2)) .+ ε, seed
end

function importance_sampler(
    weights::AbstractArray;
    ess_thresh::Float32=5f-1,
    seed::Int=1,
    )
    """
    Resample the latent variable using systematic sampling.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.

    Returns:
        The resampled indices.
        The updated seed.
    """
    
    # Systematic resampling 
    N = size(weights, 2)

    function resample(w::AbstractArray)
        ESS = 1 / sum(w.^2)
        indices = collect(1:N) 
        if ESS < ess_thresh*N
            cdf = cumsum(w)
            seed, rng = next_rng(seed)
            u0 = rand(rng) / N
            u = u0 .+ (0:N-1) ./ N
            indices = map(i -> findfirst(cdf .>= u[i]), 1:N)
            indices = reduce(vcat, indices) 
        end
        return indices
    end

    indices = map(resample, eachrow(weights))
    return indices, seed
end

function init_MoE_lkhood(
    conf::ConfParse,
    output_dim::Int;
    lkhood_seed::Int=1,
    )

    q_size = last(parse.(Int, retrieve(conf, "MIX_PRIOR", "layer_widths")))
    widths = parse.(Int, retrieve(conf, "MOE_LIKELIHOOD", "layer_widths"))
    widths = (widths..., 1)
    first(widths) !== q_size && (error("First ΩΛ_hidden_widths must be equal to the hidden dimension of the prior."))

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
    noise_var = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "llhood_noise_var"))
    gen_var = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "generator_variance"))
    lkhood_model = retrieve(conf, "MOE_LIKELIHOOD", "likelihood_model")
    output_act = retrieve(conf, "MOE_LIKELIHOOD", "output_activation")
    resampling_ess_threshold = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "resampling_ess_threshold"))

    resample_function = (weights, seed) -> @ignore_derivatives importance_sampler(weights; ess_thresh=resampling_ess_threshold, seed=seed)

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

    Ω_functions = NamedTuple()
    Λ_functions = NamedTuple()
    for i in eachindex(widths[1:end-1])
        lkhood_seed, rng = next_rng(lkhood_seed)
        base_scale = (μ_scale * (1f0 / √(Float32(widths[i])))
        .+ σ_base .* (randn(rng, Float32, widths[i], widths[i+1]) .* 2f0 .- 1f0) .* (1f0 / √(Float32(widths[i]))))
        @reset Ω_functions[Symbol("Ω_$i")] = initialize_function(widths[i], widths[i+1], base_scale)

        lkhood_seed, rng = next_rng(lkhood_seed)
        base_scale = (μ_scale * (1f0 / √(Float32(widths[i])))
        .+ σ_base .* (randn(rng, Float32, widths[i], widths[i+1]) .* 2f0 .- 1f0) .* (1f0 / √(Float32(widths[i]))))
        @reset Λ_functions[Symbol("Λ_$i")] = initialize_function(widths[i], widths[i+1], base_scale)
    end

    return MoE_lkhood(Ω_functions, Λ_functions, length(widths)-1, output_dim, noise_var, gen_var, lkhood_models[lkhood_model], activation_mapping[output_act], resample_function)
end

function Lux.initialparameters(rng::AbstractRNG, lkhood::MoE_lkhood)
    ps = NamedTuple(Symbol("Ω_$i") => Lux.initialparameters(rng, lkhood.Ω_fcns[Symbol("Ω_$i")]) for i in 1:lkhood.depth)
    for i in 1:lkhood.depth
        @reset ps[Symbol("Λ_$i")] = Lux.initialparameters(rng, lkhood.Λ_fcns[Symbol("Λ_$i")])
    end
    @reset ps[Symbol("gate_w")] = glorot_uniform(Float32, lkhood.Λ_fcns[Symbol("Λ_1")].in_dim, lkhood.out_size)
    return ps
end

function Lux.initialstates(rng::AbstractRNG, lkhood::MoE_lkhood)
    st = NamedTuple(Symbol("Ω_$i") => Lux.initialstates(rng, lkhood.Ω_fcns[Symbol("Ω_$i")]) for i in 1:lkhood.depth)
    for i in 1:lkhood.depth
        @reset st[Symbol("Λ_$i")] = Lux.initialstates(rng, lkhood.Λ_fcns[Symbol("Λ_$i")])
    end
    return st
end

end