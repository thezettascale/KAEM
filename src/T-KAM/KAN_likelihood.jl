module KAN_likelihood

export KAN_lkhood, init_KAN_lkhood, generate_from_z, importance_resampler, log_likelihood

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays, Accessors
using NNlib: sigmoid_fast, tanh_fast, relu
using ChainRules: @ignore_derivatives
using Zygote: Buffer

include("univariate_functions.jl")
include("mixture_prior.jl")
include("resamplers.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng, half_quant, full_quant, hq, fq
using .ebm_mix_prior
using .WeightResamplers

output_activation_mapping = Dict(
    "tanh" => tanh_fast,
    "sigmoid" => sigmoid_fast,
    "none" => identity,
    "step" => x -> x .> 0 |> hq,
)

lkhood_rgb = (x::AbstractArray{full_quant}, x̂::AbstractArray{full_quant}; ε=eps(full_quant)) -> -dropdims( sum( (x .- permutedims(x̂, [1, 2, 3, 5, 4])).^ 2, dims=(1,2,3) ); dims=(1,2,3) )

function lkhood_seq(x::AbstractArray{full_quant}, x̂::AbstractArray{full_quant}; ε=eps(full_quant))
    log_x̂ = log.(x̂ .+ ε)    
    ll = dropdims(sum(permutedims(log_x̂, [1, 2, 4, 3]) .* x, dims=(1,2)), dims=(1,2)) # One-hot encoded cross-entropy
    return ll ./ size(x̂, 1)
end

resampler_map = Dict(
    "residual" => residual_resampler,
    "systematic" => systematic_resampler,
    "stratified" => stratified_resampler,
)

struct KAN_lkhood <: Lux.AbstractLuxLayer
    Φ_fcns
    layernorm::Bool
    depth::Int
    out_size::Int
    σ_ε::half_quant
    σ_llhood::full_quant
    log_lkhood::Function
    output_activation::Function
    x_shape::Tuple{Vararg{Int}}
    resample_z::Function
    generate_from_z::Function
    CNN::Bool
    seq_length::Int
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
    num_samples = size(z)[end]

    # KAN functions
    for i in 1:lkhood.depth
        z = fwd(lkhood.Φ_fcns[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = dropdims(sum(z, dims=1); dims=1)

        if lkhood.layernorm && i < lkhood.depth
            z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("ln_$i")], z, ps[Symbol("ln_$i")], st[Symbol("ln_$i")])
            @ignore_derivatives @reset st[Symbol("ln_$i")] = st_new
        end
    end

    return reshape(z, lkhood.x_shape..., num_samples), st
end

# CNN generator
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
    z = reshape(z, 1, 1, size(z)...)

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

# Seq generator
function SEQ_gen(lkhood, ps, st, z::AbstractArray{half_quant})
    """
    Generate data from the Transformer decoder likelihood model.

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
    
    # Project to hidden dim 
    z = fwd(lkhood.Φ_fcns[Symbol("1")], ps[Symbol("1")], st[Symbol("1")], z)
    z = dropdims(sum(z, dims=1); dims=1)
    z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("ln_1")], z, ps[Symbol("ln_1")], st[Symbol("ln_1")])
    @ignore_derivatives @reset st[Symbol("ln_1")] = st_new
    
    # Initialize carry and first output
    carry = zeros(half_quant, size(z)) |> device
    out_z = fwd(lkhood.Φ_fcns[Symbol("3")], ps[Symbol("3")], st[Symbol("3")], z)
    out = reshape(dropdims(sum(out_z, dims=1); dims=1), lkhood.out_size, 1, size(z)[end])

    for t in 1:lkhood.seq_length-1
        z = z + carry
        carry = z
        z = fwd(lkhood.Φ_fcns[Symbol("2")], ps[Symbol("2")], st[Symbol("2")], z)
        z = dropdims(sum(z, dims=1); dims=1)

        z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("ln_2")], z, ps[Symbol("ln_2")], st[Symbol("ln_2")])
        @ignore_derivatives @reset st[Symbol("ln_2")] = st_new

        out_z = fwd(lkhood.Φ_fcns[Symbol("3")], ps[Symbol("3")], st[Symbol("3")], z)
        out_z = reshape(dropdims(sum(out_z, dims=1); dims=1), lkhood.out_size, 1, size(z)[end])
        out = cat(out, out_z, dims=2)
    end

    return out, st
end 

function log_likelihood(
    lkhood, 
    ps, 
    st, 
    x::AbstractArray{full_quant}, 
    z::AbstractArray{half_quant};
    seed::Int=1,
    ε::full_quant=eps(full_quant),
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
    Q, S, B = size(z)..., size(x)[end]

    x̂, st = lkhood.generate_from_z(lkhood, ps, st, z)

    # Add noise
    seed, rng = next_rng(seed)
    noise = lkhood.σ_ε * randn(rng, half_quant, size(x̂)..., B) |> device
    x̂ = lkhood.output_activation(x̂ .+ noise) |> fq # Accumulate across samples in full precision
    ll = lkhood.log_lkhood(x, x̂; ε=ε) ./ (2*lkhood.σ_llhood^2) 
    
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
    x_shape::Tuple{Vararg{Int}};
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

    CNN = parse(Bool, retrieve(conf, "CNN", "use_cnn_lkhood"))
    sequence_length = parse(Int, retrieve(conf, "SEQ", "sequence_length"))

    output_dim = CNN ? last(x_shape) : (sequence_length > 1 ? first(x_shape) : prod(x_shape))

    widths = (widths..., output_dim)
    first(widths) !== q_size && (error("First expert Φ_hidden_widths must be equal to the hidden dimension of the prior."))

    spline_degree = parse(Int, retrieve(conf, "KAN_LIKELIHOOD", "spline_degree"))
    layernorm = parse(Bool, retrieve(conf, "KAN_LIKELIHOOD", "layer_norm"))
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
    gen_var = parse(full_quant, retrieve(conf, "KAN_LIKELIHOOD", "generator_variance"))
    ESS_threshold = parse(full_quant, retrieve(conf, "TRAINING", "resampling_threshold_factor"))
    output_act = retrieve(conf, "KAN_LIKELIHOOD", "output_activation")
    resampler = retrieve(conf, "KAN_LIKELIHOOD", "resampler")
    verbose = parse(Bool, retrieve(conf, "TRAINING", "verbose"))
    resampler = resampler_map[resampler]

    resample_fcn = (weights, seed) -> @ignore_derivatives importance_resampler(weights; seed=seed, ESS_threshold=ESS_threshold, resampler=resampler, verbose=verbose)
    ll_model = lkhood_rgb
    generate_fcn = KAN_gen

    output_activation = sequence_length > 1 ? (x -> softmax(x, dims=1)) : get(output_activation_mapping, output_act, identity)

    Φ_functions = NamedTuple() 
    depth = length(widths)-1

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

    if CNN
        channels = parse.(Int, retrieve(conf, "CNN", "hidden_feature_dims"))
        hidden_c = (q_size, channels...)
        depth = length(hidden_c)-1
        strides = parse.(Int, retrieve(conf, "CNN", "strides"))
        k_size = parse.(Int, retrieve(conf, "CNN", "kernel_sizes"))
        paddings = parse.(Int, retrieve(conf, "CNN", "paddings"))
        act = activation_mapping[retrieve(conf, "CNN", "activation")]
        generate_fcn = CNN_gen
        layernorm = false

        length(strides) != length(hidden_c) && (error("Number of strides must be equal to the number of hidden layers + 1."))
        length(k_size) != length(hidden_c) && (error("Number of kernel sizes must be equal to the number of hidden layers + 1."))
        length(paddings) != length(hidden_c) && (error("Number of paddings must be equal to the number of hidden layers + 1."))

        for i in eachindex(hidden_c[1:end-1])
            @reset Φ_functions[Symbol("$i")] = Lux.ConvTranspose((k_size[i], k_size[i]), hidden_c[i] => hidden_c[i+1], identity; stride=strides[i], pad=paddings[i])
            @reset Φ_functions[Symbol("bn_$i")] = Lux.BatchNorm(hidden_c[i+1], relu)
        end
        @reset Φ_functions[Symbol("$(length(hidden_c))")] = Lux.ConvTranspose((k_size[end], k_size[end]), hidden_c[end] => output_dim, identity; stride=strides[end], pad=paddings[end])

    elseif sequence_length > 1
        act = activation_mapping[retrieve(conf, "SEQ", "activation")]
        hidden_dim = parse(Int, retrieve(conf, "SEQ", "hidden_dim"))
        generate_fcn = SEQ_gen
        ll_model = lkhood_seq
        layernorm = true

        # Projection layer
        lkhood_seed, rng = next_rng(lkhood_seed)
        base_scale = (μ_scale * (full_quant(1) / √(full_quant(q_size)))
        .+ σ_base .* (randn(rng, full_quant, q_size, hidden_dim) .* full_quant(2) .- full_quant(1)) .* (full_quant(1) / √(full_quant(q_size))))
        
        @reset Φ_functions[Symbol("1")] = initialize_function(q_size, hidden_dim, base_scale)
        @reset Φ_functions[Symbol("ln_1")] = Lux.LayerNorm(hidden_dim)

        # Recurrent layer
        lkhood_seed, rng = next_rng(lkhood_seed)
        base_scale = (μ_scale * (full_quant(1) / √(full_quant(hidden_dim)))
        .+ σ_base .* (randn(rng, full_quant, hidden_dim, hidden_dim) .* full_quant(2) .- full_quant(1)) .* (full_quant(1) / √(full_quant(hidden_dim))))

        @reset Φ_functions[Symbol("2")] = initialize_function(hidden_dim, hidden_dim, base_scale)
        @reset Φ_functions[Symbol("ln_2")] = Lux.LayerNorm(hidden_dim)

        # Output layer
        lkhood_seed, rng = next_rng(lkhood_seed)
        base_scale = (μ_scale * (full_quant(1) / √(full_quant(hidden_dim)))
        .+ σ_base .* (randn(rng, full_quant, hidden_dim, output_dim) .* full_quant(2) .- full_quant(1)) .* (full_quant(1) / √(full_quant(hidden_dim))))
        @reset Φ_functions[Symbol("3")] = initialize_function(hidden_dim, output_dim, base_scale)

        depth = 3
    else
        for i in eachindex(widths[1:end-1])
            lkhood_seed, rng = next_rng(lkhood_seed)
            base_scale = (μ_scale * (full_quant(1) / √(full_quant(widths[i])))
            .+ σ_base .* (randn(rng, full_quant, widths[i], widths[i+1]) .* full_quant(2) .- full_quant(1)) .* (full_quant(1) / √(full_quant(widths[i]))))
            @reset Φ_functions[Symbol("$i")] = initialize_function(widths[i], widths[i+1], base_scale)

            if (layernorm && i < depth)
                @reset Φ_functions[Symbol("ln_$i")] = Lux.LayerNorm(widths[i+1])
            end
        end
    end

    return KAN_lkhood(Φ_functions, layernorm, depth, output_dim, noise_var, gen_var, ll_model, output_activation, x_shape, resample_fcn, generate_fcn, CNN, sequence_length)
end

function Lux.initialparameters(rng::AbstractRNG, lkhood::KAN_lkhood)

    ps = NamedTuple(Symbol("$i") => Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("$i")]) for i in 1:lkhood.depth)

    if lkhood.CNN
        @reset ps[Symbol("$(lkhood.depth+1)")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("$(lkhood.depth+1)")])
        for i in 1:lkhood.depth
            @reset ps[Symbol("bn_$i")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("bn_$i")])
        end
    end

    if lkhood.layernorm 
        for i in 1:lkhood.depth-1
            @reset ps[Symbol("ln_$i")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("ln_$i")]) 
        end
    end

    return ps 
end

function Lux.initialstates(rng::AbstractRNG, lkhood::KAN_lkhood)

    st = NamedTuple(Symbol("$i") => Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("$i")]) |> hq for i in 1:lkhood.depth)

    if lkhood.CNN
        @reset st[Symbol("$(lkhood.depth+1)")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("$(lkhood.depth+1)")]) |> hq
        for i in 1:lkhood.depth
            @reset st[Symbol("bn_$i")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("bn_$i")]) |> hq
        end
    end

    if lkhood.layernorm
        for i in 1:lkhood.depth-1
            @reset st[Symbol("ln_$i")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("ln_$i")]) |> hq
        end 
    end

    return st 
end

end