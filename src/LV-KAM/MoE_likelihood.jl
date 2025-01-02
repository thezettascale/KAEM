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
    Ω_fcn::univariate_function
    Λ_fcns::NamedTuple
    γ_functions::NamedTuple
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
    z::AbstractArray;
    seed::Int=1,
    noise::Bool=true
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

    # MoE functions
    Λ, γ = copy(z), copy(z)
    for i in 1:lkhood.depth
        Λ = fwd(lkhood.Λ_fcns[Symbol("Λ_$i")], ps[Symbol("Λ_$i")], st[Symbol("Λ_$i")], Λ)
        Λ = i == 1 ? reshape(Λ, num_samples*q_size, size(Λ, 3)) : sum(Λ, dims=2)[:, 1, :]

        γ = fwd(lkhood.γ_functions[Symbol("γ_$i")], ps[Symbol("γ_$i")], st[Symbol("γ_$i")], γ)
        γ = i == 1 ? reshape(γ, num_samples*q_size, size(γ, 3)) : sum(γ, dims=2)[:, 1, :]
    end
    Λ, γ = reshape(Λ, num_samples, q_size, 1), reshape(γ, num_samples, q_size, 1)

    # Gating mechanism - Σ_q softmax(w * γ) * Λ
    w = ps[Symbol("w_gate")]
    @tullio gate[b,q,o] := γ[b,q,1] * w[o,q]
    z = softmax(gate, dims=2) .* Λ
    z = sum(z, dims=2)[:, 1, :]
    
    # Generate x
    z = fwd(lkhood.Ω_fcn, ps[Symbol("Ω")], st[Symbol("Ω")], z)[:,:,1]

    # Add noise
    seed, rng = next_rng(seed)
    ε = lkhood.σ_ε * randn(rng, Float32, size(z)) |> device
    !noise && (ε .*= 0f0)

    return lkhood.output_activation(z + ε), seed
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
    x̂, seed = generate_from_z(lkhood, ps, st, z)
    logllhood = lkhood.log_lkhood_model(
        permutedims(x[:,:,:], [2,3,1]),
        permutedims(x̂[:,:,:], [3,1,2])
    )
    return logllhood ./ (2f0*lkhood.σ_llhood^2), seed
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
    
   
    N = size(weights, 2)
    function resample(w::AbstractArray)
        ESS = 1 / sum(w.^2)
        indices = collect(1:N) 

         # Systematic resampling 
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
    q_size = (
        try 
            last(parse.(Int, retrieve(conf, "MIX_PRIOR", "layer_widths")))
        catch
            last(parse.(Int, split(retrieve(conf, "MIX_PRIOR", "layer_widths"), ",")))
        end
    )

    expert_widths = (
        try 
            parse.(Int, retrieve(conf, "MOE_LIKELIHOOD", "expert_widths"))
        catch
            parse.(Int, split(retrieve(conf, "MOE_LIKELIHOOD", "expert_widths"), ","))
        end
    )

    expert_widths = (expert_widths..., 1)
    first(expert_widths) !== q_size && (error("First expert Λ_hidden_widths must be equal to the hidden dimension of the prior."))

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
    noise_var = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "generator_noise_var"))
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

    # Output functions
    lkhood_seed, rng = next_rng(lkhood_seed)
    base_scale = (μ_scale * (1f0 / √(Float32(output_dim)))
    .+ σ_base .* (randn(rng, Float32, output_dim, 1) .* 2f0 .- 1f0) .* (1f0 / √(Float32(output_dim))))
    Ω_function = initialize_function(output_dim, 1, base_scale)

    # MoE functions
    Λ_functions = NamedTuple() # Expert functions
    γ_functions = NamedTuple() # Gating functions
    for i in eachindex(expert_widths[1:end-1])
        lkhood_seed, rng = next_rng(lkhood_seed)
        base_scale = (μ_scale * (1f0 / √(Float32(expert_widths[i])))
        .+ σ_base .* (randn(rng, Float32, expert_widths[i], expert_widths[i+1]) .* 2f0 .- 1f0) .* (1f0 / √(Float32(expert_widths[i]))))
        @reset Λ_functions[Symbol("Λ_$i")] = initialize_function(expert_widths[i], expert_widths[i+1], base_scale)
        @reset γ_functions[Symbol("γ_$i")] = initialize_function(expert_widths[i], expert_widths[i+1], base_scale)
    end

    return MoE_lkhood(Ω_function, Λ_functions, γ_functions, length(expert_widths)-1, output_dim, noise_var, gen_var, lkhood_models[lkhood_model], activation_mapping[output_act], resample_function)
end

function Lux.initialparameters(rng::AbstractRNG, lkhood::MoE_lkhood)
    ps = NamedTuple()
    for i in 1:lkhood.depth
        @reset ps[Symbol("Λ_$i")] = Lux.initialparameters(rng, lkhood.Λ_fcns[Symbol("Λ_$i")])
        @reset ps[Symbol("γ_$i")] = Lux.initialparameters(rng, lkhood.γ_functions[Symbol("γ_$i")])
    end
    @reset ps[Symbol("Ω")] = Lux.initialparameters(rng, lkhood.Ω_fcn)
    @reset ps[Symbol("w_gate")] = glorot_normal(Float32, lkhood.out_size, lkhood.Λ_fcns[Symbol("Λ_1")].in_dim)
    return ps
end

function Lux.initialstates(rng::AbstractRNG, lkhood::MoE_lkhood)
    st = NamedTuple()
    for i in 1:lkhood.depth
        @reset st[Symbol("Λ_$i")] = Lux.initialstates(rng, lkhood.Λ_fcns[Symbol("Λ_$i")])
        @reset st[Symbol("γ_$i")] = Lux.initialstates(rng, lkhood.γ_functions[Symbol("γ_$i")])
    end
    @reset st[Symbol("Ω")] = Lux.initialstates(rng, lkhood.Ω_fcn)
    return st
end

end