module KAN_likelihood

export KAN_lkhood, init_KAN_lkhood, generate_from_z, importance_resampler, log_likelihood

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays, Accessors
using NNlib: sigmoid_fast, tanh_fast, relu
using ChainRules: @ignore_derivatives

include("univariate_functions.jl")
include("mixture_prior.jl")
include("resamplers.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng, half_quant, full_quant
using .ebm_mix_prior
using .WeightResamplers

const CNN_hq = half_quant == Float16 ? Lux.f16 : Lux.f32

output_activation_mapping = Dict(
    "tanh" => tanh_fast,
    "sigmoid" => sigmoid_fast,
    "none" => identity
)

lkhood_models_flat = Dict(
    "l2" => (x::AbstractArray{half_quant}, x̂::AbstractArray{half_quant}; ε=eps(half_quant)) -> -dropdims( sum( (x' .- permutedims(x̂, [3, 2, 1])).^2 ; dims=2 ); dims=2 ),  
    "bernoulli" => (x::AbstractArray{half_quant}, x̂::AbstractArray{half_quant}; ε=eps(half_quant)) -> dropdims( sum( x .* log(permutedims(x̂, [3, 2, 1]) .+ ε) .+ (1 .- x) .* log(1 .- permutedims(x̂, [3, 2, 1]) .+ ε) ; dims=2 ); dims=2 ),
)

lkhood_model_rgb = (x::AbstractArray{half_quant}, x̂::AbstractArray{half_quant}; ε=eps(half_quant)) -> -dropdims( sum( (x .- permutedims(x̂, [1,2,3,5,4])).^2 ; dims=(1,2,3) ); dims=(1,2,3) ) 

llhoods_dict = Dict(
    false => lkhood_models_flat,
    true => Dict(
        "l2" => lkhood_model_rgb,
    )
)

resampler_map = Dict(
    "residual" => residual_resampler,
    "systematic" => systematic_resampler,
    "stratified" => stratified_resampler,
)

struct KAN_lkhood <: Lux.AbstractLuxLayer
    Φ_fcns::NamedTuple
    depth::Int
    out_size::Int
    σ_ε::half_quant
    σ_llhood::half_quant
    log_lkhood_model::Function
    output_activation::Function
    resample_z::Function
    generate_from_z::Function
    CNN::Bool
end

function KAN_gen(
    lkhood, 
    ps, 
    st, 
    z::AbstractArray{half_quant}    
    )
    """
    Generate data from the KAN likelihood model.

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

    return z, st
end

function CNN_gen(
    lkhood, 
    ps, 
    st, 
    z::AbstractArray{half_quant}    
    )
    """
    Generate data from the CNN likelihood model.

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
    z = reshape(z, 1, 1, q_size, num_samples)

    for i in 1:lkhood.depth
        z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("$i")], z, ps[Symbol("$i")], st[Symbol("$i")])
        @ignore_derivatives @reset st[Symbol("$i")] = st_new    

        z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("bn_$i")], z, ps[Symbol("bn_$i")], st[Symbol("bn_$i")])
        @ignore_derivatives @reset st[Symbol("bn_$i")] = st_new
    end

    z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("$(lkhood.depth+1)")], z, ps[Symbol("$(lkhood.depth+1)")], st[Symbol("$(lkhood.depth+1)")])
    @reset st[Symbol("$(lkhood.depth+1)")] = st_new

    return z, st
end

function log_likelihood(
    lkhood, 
    ps, 
    st, 
    x::AbstractArray{half_quant}, 
    z::AbstractArray{half_quant};
    full_precision::Bool=false,
    seed::Int=1,
    ε::half_quant=eps(half_quant)
    )
    """
    Evaluate the unnormalized log-likelihood of the KAN generator.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        tempered: Whether to use tempered likelihood.
        seed: The seed for the random number generator.

    Returns:
        The unnormalized log-likelihood.
        The updated seed.
    """
    S, Q, B = size(z)..., size(x)[end]

    x̂, st = lkhood.generate_from_z(lkhood, ps, st, z)

    # Add noise
    seed, rng = next_rng(seed)
    noise = lkhood.σ_ε * randn(rng, half_quant, size(x̂)..., B) |> device
    x̂ = lkhood.output_activation(x̂ .+ noise)
    ll = lkhood.log_lkhood_model(x, x̂; ε=ε) ./ (2*lkhood.σ_llhood^2)
    
    # Loss unstable if accumulated in half precision, grads are fine though
    @ignore_derivatives if full_precision
        ll = full_quant.(ll)
    end
    
    return ll, st, seed
end

function importance_resampler(
    weights::AbstractArray{full_quant};
    seed::Int=1,
    ESS_threshold::full_quant=full_quant(0.5),
    resampler::Function=systematic_sampler,
    verbose::Bool=false,
)
    """
    Filter the latent variable for a index of the Steppingstone sum using residual resampling.

    Args:
        logllhood: A matrix of log-likelihood values.
        weights: The weights of the population.
        t_resample: The temperature at which the last resample occurred.
        t2: The temperature at which to update the weights.
        seed: Random seed for reproducibility.
        ESS_threshold: The threshold for the effective sample size.
        resampler: The resampling function.

    Returns:
        - The resampled indices.
        - The updated seed.
    """
    B, N = size(weights)

    # Check effective sample size
    ESS = dropdims(1 ./ sum(weights.^2, dims=2); dims=2)
    ESS_bool = ESS .< ESS_threshold*N
    
    # Only resample when needed 
    verbose && (any(ESS_bool) && println("Resampling!"))
    any(ESS_bool) && return resampler(cpu_device()(weights), cpu_device()(ESS_bool), B, N; seed=seed)
    return repeat(collect(1:N)', B, 1), seed
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

    widths = (
        try 
            parse.(Int, retrieve(conf, "KAN_LIKELIHOOD", "widths"))
        catch
            parse.(Int, split(retrieve(conf, "KAN_LIKELIHOOD", "widths"), ","))
        end
    )

    widths = (widths..., output_dim)
    first(widths) !== q_size && (error("First expert Φ_hidden_widths must be equal to the hidden dimension of the prior."))

    spline_degree = parse(Int, retrieve(conf, "KAN_LIKELIHOOD", "spline_degree"))
    base_activation = retrieve(conf, "KAN_LIKELIHOOD", "base_activation")
    spline_function = retrieve(conf, "KAN_LIKELIHOOD", "spline_function")
    grid_size = parse(Int, retrieve(conf, "KAN_LIKELIHOOD", "grid_size"))
    grid_update_ratio = parse(half_quant, retrieve(conf, "KAN_LIKELIHOOD", "grid_update_ratio"))
    grid_range = parse.(half_quant, retrieve(conf, "KAN_LIKELIHOOD", "grid_range"))
    ε_scale = parse(half_quant, retrieve(conf, "KAN_LIKELIHOOD", "ε_scale"))
    μ_scale = parse(full_quant, retrieve(conf, "KAN_LIKELIHOOD", "μ_scale"))
    σ_base = parse(full_quant, retrieve(conf, "KAN_LIKELIHOOD", "σ_base"))
    σ_spline = parse(full_quant, retrieve(conf, "KAN_LIKELIHOOD", "σ_spline"))
    init_τ = parse(full_quant, retrieve(conf, "KAN_LIKELIHOOD", "init_τ"))
    τ_trainable = parse(Bool, retrieve(conf, "KAN_LIKELIHOOD", "τ_trainable"))
    τ_trainable = spline_function == "B-spline" ? false : τ_trainable
    noise_var = parse(half_quant, retrieve(conf, "KAN_LIKELIHOOD", "generator_noise_var"))
    gen_var = parse(half_quant, retrieve(conf, "KAN_LIKELIHOOD", "generator_variance"))
    ESS_threshold = parse(full_quant, retrieve(conf, "TRAINING", "resampling_threshold_factor"))
    output_act = retrieve(conf, "KAN_LIKELIHOOD", "output_activation")
    resampler = retrieve(conf, "KAN_LIKELIHOOD", "resampler")
    verbose = parse(Bool, retrieve(conf, "TRAINING", "verbose"))
    resampler = resampler_map[resampler]

    resample_fcn = (weights, seed) -> @ignore_derivatives importance_resampler(weights; seed=seed, ESS_threshold=ESS_threshold, resampler=resampler, verbose=verbose)

    CNN = parse(Bool, retrieve(conf, "CNN", "use_cnn_lkhood"))

    lkhood_model = CNN ? "l2" : retrieve(conf, "KAN_LIKELIHOOD", "likelihood_model")
    ll_model = llhoods_dict[CNN][lkhood_model]
    generate_fcn = CNN ? CNN_gen : KAN_gen

    Φ_functions = NamedTuple() 
    depth = length(widths)-1

    if !CNN
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
            init_τ=init_τ,
            τ_trainable=τ_trainable,
        )

        for i in eachindex(widths[1:end-1])
            lkhood_seed, rng = next_rng(lkhood_seed)
            base_scale = (μ_scale * (full_quant(1) / √(full_quant(widths[i])))
            .+ σ_base .* (randn(rng, full_quant, widths[i], widths[i+1]) .* full_quant(2) .- full_quant(1)) .* (full_quant(1) / √(full_quant(widths[i]))))
            @reset Φ_functions[Symbol("$i")] = initialize_function(widths[i], widths[i+1], base_scale)
        end
    else
        channels = parse.(Int, retrieve(conf, "CNN", "hidden_feature_dims"))
        hidden_c = (q_size, channels...)
        depth = length(hidden_c)-1
        strides = parse.(Int, retrieve(conf, "CNN", "strides"))
        k_size = parse.(Int, retrieve(conf, "CNN", "kernel_sizes"))
        paddings = parse.(Int, retrieve(conf, "CNN", "paddings"))

        length(strides) != length(hidden_c) && (error("Number of strides must be equal to the number of hidden layers + 1."))
        length(k_size) != length(hidden_c) && (error("Number of kernel sizes must be equal to the number of hidden layers + 1."))
        length(paddings) != length(hidden_c) && (error("Number of paddings must be equal to the number of hidden layers + 1."))

        for i in eachindex(hidden_c[1:end-1])
            @reset Φ_functions[Symbol("$i")] = Lux.ConvTranspose((k_size[i], k_size[i]), hidden_c[i] => hidden_c[i+1], identity; stride=strides[i], pad=paddings[i])
            @reset Φ_functions[Symbol("bn_$i")] = Lux.BatchNorm(hidden_c[i+1], leakyrelu)
        end
        @reset Φ_functions[Symbol("$(length(hidden_c))")] = Lux.ConvTranspose((k_size[end], k_size[end]), hidden_c[end] => output_dim, identity; stride=strides[end], pad=paddings[end]) 
    end

    return KAN_lkhood(Φ_functions, depth, output_dim, noise_var, gen_var, ll_model, output_activation_mapping[output_act], resample_fcn, generate_fcn, CNN)
end

function Lux.initialparameters(rng::AbstractRNG, lkhood::KAN_lkhood)
    ps = NamedTuple(Symbol("$i") => Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("$i")]) for i in 1:lkhood.depth)
    if lkhood.CNN
        @reset ps[Symbol("$(lkhood.depth+1)")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("$(lkhood.depth+1)")])
        for i in 1:lkhood.depth
            @reset ps[Symbol("bn_$i")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("bn_$i")])
        end
    end
    return ps
end

function Lux.initialstates(rng::AbstractRNG, lkhood::KAN_lkhood)
    st = NamedTuple(Symbol("$i") => Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("$i")]) for i in 1:lkhood.depth)
    
    if lkhood.CNN
        @reset st[Symbol("$(lkhood.depth+1)")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("$(lkhood.depth+1)")])
        for i in 1:lkhood.depth
            @reset st[Symbol("bn_$i")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("bn_$i")]) |> CNN_hq
        end
    end
    return st
end

end