module KAN_likelihood

export KAN_lkhood, init_KAN_lkhood, log_likelihood, generate_from_z, systematic_sampler, particle_filter

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays, Accessors
using NNlib: sigmoid_fast, tanh_fast
using ChainRules: @ignore_derivatives

include("univariate_functions.jl")
include("mixture_prior.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng, quant
using .ebm_mix_prior

output_activation_mapping = Dict(
    "tanh" => tanh_fast,
    "sigmoid" => sigmoid_fast,
    "none" => identity
)

lkhood_models = Dict(
    "l2" => (x::AbstractArray{quant}, x̂::AbstractArray{quant}) -> -dropdims(sum((x .- x̂).^2, dims=3), dims=3),
    "bernoulli" => (x::AbstractArray{quant}, x̂::AbstractArray{quant}) -> dropdims(sum(x .* log.(x̂ .+ eps(eltype(x))) .+ (1 .- x) .* log.(1 .- x̂ .+ eps(eltype(x))), dims=3), dims=3),
)

struct KAN_lkhood <: Lux.AbstractLuxLayer
    Φ_fcns::NamedTuple
    depth::Int
    out_size::Int
    σ_ε::quant
    σ_llhood::quant
    log_lkhood_model::Function
    output_activation::Function
    pf_resample::Function
end

function generate_from_z(
    lkhood::KAN_lkhood, 
    ps, 
    st, 
    z::AbstractArray{quant};
    seed::Int=1,
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
    ε = lkhood.σ_ε * randn(rng, quant, size(z)) |> device
    return lkhood.output_activation(z + ε), seed
end

function log_likelihood(
    lkhood::KAN_lkhood, 
    ps, 
    st, 
    x::AbstractArray{quant},
    z::AbstractArray{quant};
    seed::Int=1,
    )
    """
    Evaluate the log-likelihood of the data given the latent variable.
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
    return logllhood ./ (quant(2)*lkhood.σ_llhood^2), seed
end

function particle_filter(
    logllhood::AbstractArray{quant},
    t::quant;
    seed::Int=1,
)
    """
    Filter the latent variable for a index of the Steppingstone sum using residual resampling.

    Args:
        logllhood: A matrix of log-likelihood values.
        Δt: The change in temperature.
        seed: Random seed for reproducibility.

    Returns:
        - The resampled indices.
        - The resampled log-likelihoods.
        - The updated seed.
    """
    B, N = size(logllhood)

    # Update the weights to the next temperature
    weights = softmax(t .* logllhood, dims=2)

    # Number times to replicate each particle
    integer_counts = floor.(weights .* N) .|> Int
    num_remaining = dropdims(N .- sum(integer_counts, dims=2); dims=2)

    # Residual weights to resample from
    residual_weights = softmax(weights .- (integer_counts ./ N); dims=2)

    # CDF and variate for resampling
    seed, rng = next_rng(seed)
    u = rand(rng, quant, B, maximum(num_remaining))
    cdf = cumsum(residual_weights, dims=2)

    idxs = Array{Int}(undef, B, N)
    Threads.@threads for b in 1:B
        c = 1

        # Deterministic replication
        for s in 1:N
            count = integer_counts[b, s]
            if count > 0
                idxs[b, c:c+count-1] .= s
                c += count
            end
        end

        # Multinomial resampling
        if num_remaining[b] > 0
            idxs[b, c:end] .= searchsortedfirst.(Ref(cdf[b, :]), u[b, 1:num_remaining[b]])
        end
    end
    replace!(idxs, N+1 => N)
    
    return idxs, logllhood[idxs], seed
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
    grid_update_ratio = parse(quant, retrieve(conf, "KAN_LIKELIHOOD", "grid_update_ratio"))
    grid_range = parse.(quant, retrieve(conf, "KAN_LIKELIHOOD", "grid_range"))
    ε_scale = parse(quant, retrieve(conf, "KAN_LIKELIHOOD", "ε_scale"))
    μ_scale = parse(quant, retrieve(conf, "KAN_LIKELIHOOD", "μ_scale"))
    σ_base = parse(quant, retrieve(conf, "KAN_LIKELIHOOD", "σ_base"))
    σ_spline = parse(quant, retrieve(conf, "KAN_LIKELIHOOD", "σ_spline"))
    init_η = parse(quant, retrieve(conf, "KAN_LIKELIHOOD", "init_η"))
    η_trainable = parse(Bool, retrieve(conf, "KAN_LIKELIHOOD", "η_trainable"))
    η_trainable = spline_function == "B-spline" ? false : η_trainable
    noise_var = parse(quant, retrieve(conf, "KAN_LIKELIHOOD", "generator_noise_var"))
    gen_var = parse(quant, retrieve(conf, "KAN_LIKELIHOOD", "generator_variance"))
    lkhood_model = retrieve(conf, "KAN_LIKELIHOOD", "likelihood_model")
    output_act = retrieve(conf, "KAN_LIKELIHOOD", "output_activation")

    resample_fcn = (logllhood, t, seed) -> @ignore_derivatives particle_filter(logllhood, t; seed=seed)

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
        base_scale = (μ_scale * (quant(1) / √(quant(expert_widths[i])))
        .+ σ_base .* (randn(rng, quant, expert_widths[i], expert_widths[i+1]) .* quant(2) .- quant(1)) .* (quant(1) / √(quant(expert_widths[i]))))
        @reset Φ_functions[Symbol("$i")] = initialize_function(expert_widths[i], expert_widths[i+1], base_scale)
    end

    return KAN_lkhood(Φ_functions, length(expert_widths)-1, output_dim, noise_var, gen_var, lkhood_models[lkhood_model], output_activation_mapping[output_act], resample_fcn)
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