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
using .Utils: device, next_rng, half_quant, full_quant
using .ebm_mix_prior
using .WeightResamplers

const hq = half_quant == Float16 ? Lux.f16 : Lux.f32
const fq = full_quant == Float16 ? Lux.f16 : (full_quant == Float64 ? Lux.f64 : Lux.f32)

output_activation_mapping = Dict(
    "tanh" => tanh_fast,
    "sigmoid" => sigmoid_fast,
    "none" => identity
)

lkhood_models_flat = Dict(
    "l2" => (x::AbstractArray{full_quant}, x̂::AbstractArray{full_quant}; ε=eps(full_quant)) -> -dropdims( sum( @tullio(out[b, s, o] := (x[o, b] - x̂[s, o, b])^2) ; dims=3 ); dims=3 ), 
    "bernoulli" => (x::AbstractArray{full_quant}, x̂::AbstractArray{full_quant}; ε=eps(full_quant)) -> dropdims( sum( @tullio(out[b, s, o] := x[o, b] * log(x̂[s, o, b] + ε) + (1 - x[o, b]) * log(1 - x̂[s, o, b] + ε)) ; dims=3 ); dims=3 ),
)

lkhood_model_rgb = (x::AbstractArray{full_quant}, x̂::AbstractArray{full_quant}; ε=eps(full_quant)) -> -dropdims( sum( @tullio(out[b, s, h, w, c] := (x[h, w, c, b] - x̂[h, w, c, s, b])^2) ; dims=(3,4,5) ); dims=(3,4,5) )

function lkhood_model_seq(x::AbstractArray{full_quant}, x̂::AbstractArray{full_quant}; ε=eps(full_quant))
    log_x̂ = log.(x̂ .+ ε)    
    @tullio ll[b,s] := log_x̂[v,t,s,b] * x[v,t,b] # One-hot mask
    return ll ./ size(x̂, 1)
end

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
    Φ_fcns
    depth::Int
    out_size::Int
    σ_ε::half_quant
    σ_llhood::full_quant
    log_lkhood_model::Function
    output_activation::Function
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
    num_samples, q_size = size(z)

    # KAN functions
    for i in 1:lkhood.depth
        z = fwd(lkhood.Φ_fcns[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = dropdims(sum(z, dims=2); dims=2)
    end

    return z, st
end

# CNN and LSTM generators
CNN_gen = (lkhood, ps, st, z::AbstractArray{half_quant}) -> Lux.apply(lkhood.Φ_fcns, reshape(z, 1, 1, size(z, 2), size(z, 1)), ps, st)
SEQ_gen = (lkhood, ps, st, z::AbstractArray{half_quant}) -> Lux.apply(lkhood.Φ_fcns, repeat(reshape(z, size(z,2), 1, size(z,1)), 1, lkhood.seq_length, 1), ps, st)

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
    S, Q, B = size(z)..., size(x)[end]

    x̂, st = lkhood.generate_from_z(lkhood, ps, st, z)

    # Add noise
    seed, rng = next_rng(seed)
    noise = lkhood.σ_ε * randn(rng, half_quant, size(x̂)..., B) |> device
    x̂ = lkhood.output_activation(x̂ .+ noise) |> fq
    ll = lkhood.log_lkhood_model(x, x̂; ε=ε) ./ (2*lkhood.σ_llhood^2) 
    
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

    CNN = parse(Bool, retrieve(conf, "CNN", "use_cnn_lkhood"))
    sequence_length = parse(Int, retrieve(conf, "LSTM", "sequence_length"))

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
    gen_var = parse(full_quant, retrieve(conf, "KAN_LIKELIHOOD", "generator_variance"))
    ESS_threshold = parse(full_quant, retrieve(conf, "TRAINING", "resampling_threshold_factor"))
    output_act = retrieve(conf, "KAN_LIKELIHOOD", "output_activation")
    resampler = retrieve(conf, "KAN_LIKELIHOOD", "resampler")
    verbose = parse(Bool, retrieve(conf, "TRAINING", "verbose"))
    resampler = resampler_map[resampler]

    resample_fcn = (weights, seed) -> @ignore_derivatives importance_resampler(weights; seed=seed, ESS_threshold=ESS_threshold, resampler=resampler, verbose=verbose)

    lkhood_model = CNN ? "l2" : retrieve(conf, "KAN_LIKELIHOOD", "likelihood_model")
    ll_model = sequence_length > 1 ? lkhood_model_seq : llhoods_dict[CNN][lkhood_model]
    generate_fcn = CNN ? CNN_gen : KAN_gen
    generate_fcn = sequence_length > 1 ? SEQ_gen : generate_fcn
    output_activation = sequence_length > 1 ? (x -> softmax(x, dims=1)) : output_activation_mapping[output_act]

    Φ_functions = NamedTuple() 
    depth = length(widths)-1

    if CNN
        channels = parse.(Int, retrieve(conf, "CNN", "hidden_feature_dims"))
        hidden_c = (q_size, channels...)
        depth = length(hidden_c)-1
        strides = parse.(Int, retrieve(conf, "CNN", "strides"))
        k_size = parse.(Int, retrieve(conf, "CNN", "kernel_sizes"))
        paddings = parse.(Int, retrieve(conf, "CNN", "paddings"))
        act = activation_mapping[retrieve(conf, "CNN", "activation")]

        length(strides) != length(hidden_c) && (error("Number of strides must be equal to the number of hidden layers + 1."))
        length(k_size) != length(hidden_c) && (error("Number of kernel sizes must be equal to the number of hidden layers + 1."))
        length(paddings) != length(hidden_c) && (error("Number of paddings must be equal to the number of hidden layers + 1."))

        Φ_functions = Lux.Chain(
        Iterators.flatten(
            ((Lux.ConvTranspose((k_size[i], k_size[i]), c => hidden_c[i+1], identity; stride=strides[i], pad=paddings[i]),
            Lux.BatchNorm(hidden_c[i+1], act)
            ) for (i, c) in enumerate(hidden_c[1:end-1]))
        )..., 
        Lux.ConvTranspose((k_size[end], k_size[end]), hidden_c[end] => output_dim, identity; stride=strides[end], pad=paddings[end])
        )

    elseif sequence_length > 1
        act = activation_mapping[retrieve(conf, "LSTM", "activation")]
        hidden_dim = parse(Int, retrieve(conf, "LSTM", "hidden_dim"))

        Φ_functions = Lux.Chain(
            Lux.Dense(q_size => hidden_dim, act),
            Lux.Recurrence(
                Lux.LSTMCell(hidden_dim => hidden_dim),
                return_sequence=true
            ),
            x -> hq(x), # LSTM stubbornly returns f32
            x -> reduce(hcat, map(z -> permutedims(z[:,:,:], (1,3,2)), x)),
            Lux.Dense(hidden_dim => output_dim)
        )

    else
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
    end

    return KAN_lkhood(Φ_functions, depth, output_dim, noise_var, gen_var, ll_model, output_activation, resample_fcn, generate_fcn, CNN, sequence_length)
end

function Lux.initialparameters(rng::AbstractRNG, lkhood::KAN_lkhood)
    ps = (
        lkhood.seq_length > 1 || lkhood.CNN ? 
        Lux.initialparameters(rng, lkhood.Φ_fcns) : 
        NamedTuple(Symbol("$i") => Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("$i")]) for i in 1:lkhood.depth)
    )
    return ps
end

function Lux.initialstates(rng::AbstractRNG, lkhood::KAN_lkhood)
    st = (
        lkhood.seq_length > 1 || lkhood.CNN ?
        Lux.initialstates(rng, lkhood.Φ_fcns) |> hq :
        NamedTuple(Symbol("$i") => Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("$i")]) for i in 1:lkhood.depth)
    )
    return st
end

end