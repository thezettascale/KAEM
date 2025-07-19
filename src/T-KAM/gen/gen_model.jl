module GeneratorModel

export GenModel,
    init_GenModel,
    generate_from_z,
    importance_resampler,
    log_likelihood_IS,
    log_likelihood_MALA

using CUDA, KernelAbstractions
using ConfParser,
    Random, Lux, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays, Accessors
using NNlib: sigmoid_fast, tanh_fast, relu, gelu, sigmoid, tanh

include("../kan/univariate_functions.jl")
include("../ebm/ebm_model.jl")
include("resamplers.jl")
include("generator_fcns.jl")
include("loglikelihoods.jl")
include("../../utils.jl")
using .UnivariateFunctions
using .Utils: device, half_quant, full_quant, hq, fq
using .EBM_Model
using .WeightResamplers
using .GeneratorFCNs
using .LogLikelihoods

const output_activation_mapping =
    Dict("tanh" => tanh_fast, "sigmoid" => sigmoid_fast, "none" => identity)

const resampler_map = Dict(
    "residual" => residual_resampler,
    "systematic" => systematic_resampler,
    "stratified" => stratified_resampler,
)

struct GenModel{T<:half_quant} <: Lux.AbstractLuxLayer
    Φ_fcns::Tuple
    layernorms::Tuple
    batchnorms::Tuple
    attention::NamedTuple
    layernorm_bool::Bool
    batchnorm_bool::Bool
    depth::Int
    out_size::Int
    σ_llhood::T
    output_activation::Function
    x_shape::Tuple{Vararg{Int}}
    resample_z::Function
    generate_from_z::Function
    CNN::Bool
    seq_length::Int
    d_model::Int
end

function init_GenModel(
    conf::ConfParse,
    x_shape::Tuple{Vararg{Int}};
    rng::AbstractRNG = Random.default_rng(),
)

    prior_widths = (
        try
            parse.(Int, retrieve(conf, "EbmModel", "layer_widths"))
        catch
            parse.(Int, split(retrieve(conf, "EbmModel", "layer_widths"), ","))
        end
    )

    q_size = length(prior_widths) > 2 ? first(prior_widths) : last(prior_widths)

    widths = (
        try
            parse.(Int, retrieve(conf, "GeneratorModel", "widths"))
        catch
            parse.(Int, split(retrieve(conf, "GeneratorModel", "widths"), ","))
        end
    )

    CNN = parse(Bool, retrieve(conf, "CNN", "use_cnn_lkhood"))
    sequence_length = parse(Int, retrieve(conf, "SEQ", "sequence_length"))

    output_dim =
        CNN ? last(x_shape) : (sequence_length > 1 ? first(x_shape) : prod(x_shape))

    widths = (widths..., output_dim)
    first(widths) !== q_size && (error(
        "First expert Φ_hidden_widths must be equal to the hidden dimension of the prior.",
        widths,
        " != ",
        q_size,
    ))

    spline_degree = parse(Int, retrieve(conf, "GeneratorModel", "spline_degree"))
    layernorm_bool = parse(Bool, retrieve(conf, "GeneratorModel", "layer_norm"))
    base_activation = retrieve(conf, "GeneratorModel", "base_activation")
    spline_function = retrieve(conf, "GeneratorModel", "spline_function")
    grid_size = parse(Int, retrieve(conf, "GeneratorModel", "grid_size"))
    grid_update_ratio =
        parse(half_quant, retrieve(conf, "GeneratorModel", "grid_update_ratio"))
    grid_range = parse.(half_quant, retrieve(conf, "GeneratorModel", "grid_range"))
    ε_scale = parse(half_quant, retrieve(conf, "GeneratorModel", "ε_scale"))
    μ_scale = parse(full_quant, retrieve(conf, "GeneratorModel", "μ_scale"))
    σ_base = parse(full_quant, retrieve(conf, "GeneratorModel", "σ_base"))
    σ_spline = parse(full_quant, retrieve(conf, "GeneratorModel", "σ_spline"))
    init_τ = parse(full_quant, retrieve(conf, "GeneratorModel", "init_τ"))
    τ_trainable = parse(Bool, retrieve(conf, "GeneratorModel", "τ_trainable"))
    τ_trainable = spline_function == "B-spline" ? false : τ_trainable
    gen_var = parse(half_quant, retrieve(conf, "GeneratorModel", "generator_variance"))
    ESS_threshold =
        parse(full_quant, retrieve(conf, "TRAINING", "resampling_threshold_factor"))
    output_act = retrieve(conf, "GeneratorModel", "output_activation")
    resampler = retrieve(conf, "GeneratorModel", "resampler")
    verbose = parse(Bool, retrieve(conf, "TRAINING", "verbose"))
    resampler = get(resampler_map, resampler, systematic_resampler)
    batchnorm_bool = false

    resample_fcn =
        (weights, rng) -> importance_resampler(
            weights;
            rng = rng,
            ESS_threshold = ESS_threshold,
            resampler = resampler,
            verbose = verbose,
        )
    generate_fcn = KAN_fwd

    output_activation =
        sequence_length > 1 ? (x -> softmax(x, dims = 1)) :
        get(output_activation_mapping, output_act, identity)

    depth = length(widths)-1
    d_model = 0

    initialize_function =
        (in_dim, out_dim, base_scale) -> init_function(
            in_dim,
            out_dim;
            spline_degree = spline_degree,
            base_activation = base_activation,
            spline_function = spline_function,
            grid_size = grid_size,
            grid_update_ratio = grid_update_ratio,
            grid_range = Tuple(grid_range),
            ε_scale = ε_scale,
            σ_base = base_scale,
            σ_spline = σ_spline,
            init_τ = init_τ,
            τ_trainable = τ_trainable,
        )

    fcns_temp = []
    layernorms_temp = []
    batchnorms_temp = []

    Φ_functions = ()
    layernorms = ()
    batchnorms = ()
    attention = NamedTuple()

    if CNN
        channels = parse.(Int, retrieve(conf, "CNN", "hidden_feature_dims"))
        hidden_c = (q_size, channels...)
        depth = length(hidden_c)-1
        strides = parse.(Int, retrieve(conf, "CNN", "strides"))
        k_size = parse.(Int, retrieve(conf, "CNN", "kernel_sizes"))
        paddings = parse.(Int, retrieve(conf, "CNN", "paddings"))
        act = activation_mapping[retrieve(conf, "CNN", "activation")]
        batchnorm_bool = parse(Bool, retrieve(conf, "CNN", "batchnorm"))
        generate_fcn = CNN_fwd
        layernorm_bool = false

        length(strides) != length(hidden_c) &&
            (error("Number of strides must be equal to the number of hidden layers + 1."))
        length(k_size) != length(hidden_c) && (error(
            "Number of kernel sizes must be equal to the number of hidden layers + 1.",
        ))
        length(paddings) != length(hidden_c) &&
            (error("Number of paddings must be equal to the number of hidden layers + 1."))

        for i in eachindex(hidden_c[1:(end-1)])
            push!(fcns_temp, Lux.ConvTranspose(
                (k_size[i], k_size[i]),
                hidden_c[i] => hidden_c[i+1],
                identity;
                stride = strides[i],
                pad = paddings[i],
            ))
            if batchnorm_bool
                push!(batchnorms_temp, Lux.BatchNorm(hidden_c[i+1], act))
            end
        end
        push!(fcns_temp, Lux.ConvTranspose(
            (k_size[end], k_size[end]),
            hidden_c[end] => output_dim,
            identity;
            stride = strides[end],
            pad = paddings[end],
        ))

        Φ_functions = ntuple(i -> fcns_temp[i], length(hidden_c)-1)
        batchnorms = ntuple(i -> batchnorms_temp[i], length(hidden_c)-1)

    elseif sequence_length > 1

        act = gelu
        generate_fcn = SEQ_fwd

        # Single block Transformer decoder
        d_model = parse(Int, retrieve(conf, "SEQ", "d_model"))

        # Projection
        push!(fcns_temp, Lux.Dense(q_size => d_model))
        push!(layernorms_temp, Lux.LayerNorm((d_model, 1), gelu))

        # Query, Key, Value
        attention = (
            Q = Lux.Dense(d_model => d_model),
            K = Lux.Dense(d_model => d_model),
            V = Lux.Dense(d_model => d_model),
        )

        # Feed forward
        push!(fcns_temp, Lux.Dense(d_model => d_model))
        push!(layernorms_temp, Lux.LayerNorm((d_model, 1), gelu))

        # Output layer
        push!(fcns_temp, Lux.Dense(d_model => output_dim))
        depth = 3

        Φ_functions = ntuple(i -> fcns_temp[i], 3)
        layernorms = ntuple(i -> layernorms_temp[i], 2)
    else
        for i in eachindex(widths[1:(end-1)])
            base_scale = (
                μ_scale * (one(full_quant) / √(full_quant(widths[i]))) .+
                σ_base .* (
                    randn(rng, full_quant, widths[i], widths[i+1]) .* full_quant(2) .-
                    one(full_quant)
                ) .* (one(full_quant) / √(full_quant(widths[i])))
            )
            push!(fcns_temp, initialize_function(widths[i], widths[i+1], base_scale))

            if (layernorm_bool && i < depth)
                push!(layernorms_temp, Lux.LayerNorm(widths[i+1]))
            end
        end

        Φ_functions = ntuple(i -> fcns_temp[i], depth)
        if layernorm_bool && length(layernorms_temp) > 0
            layernorms = ntuple(i -> layernorms_temp[i], depth-1)
        end
    end

    return GenModel(
        Φ_functions,
        layernorms,
        batchnorms,
        attention,
        layernorm_bool,
        batchnorm_bool,
        depth,
        output_dim,
        gen_var,
        output_activation,
        x_shape,
        resample_fcn,
        generate_fcn,
        CNN,
        sequence_length,
        d_model,
    )
end

function Lux.initialparameters(rng::AbstractRNG, lkhood::GenModel{T}) where {T<:half_quant}
    fcn_ps = ntuple(i -> Lux.initialparameters(rng, lkhood.Φ_fcns[i]), lkhood.depth)
    layernorm_ps = ()
    if lkhood.layernorm_bool && length(lkhood.layernorms) > 0
        layernorm_ps = ntuple(i -> Lux.initialparameters(rng, lkhood.layernorms[i]), lkhood.depth-1)
    end

    batchnorm_ps = ()
    if lkhood.batchnorm_bool && length(lkhood.batchnorms) > 0
        batchnorm_ps = ntuple(i -> Lux.initialparameters(rng, lkhood.batchnorms[i]), lkhood.depth-1)
    end

    attention_ps = ()
    if lkhood.seq_length > 1
        attention_ps = (
            Q = Lux.initialparameters(rng, lkhood.attention.Q),
            K = Lux.initialparameters(rng, lkhood.attention.K),
            V = Lux.initialparameters(rng, lkhood.attention.V),
        )
    end

    return (
        fcn = fcn_ps,
        layernorm = layernorm_ps,
        batchnorm = batchnorm_ps,
        attention = attention_ps,
    )
end

function Lux.initialstates(rng::AbstractRNG, lkhood::GenModel{T}) where {T<:half_quant}
    fcn_st = ntuple(i -> Lux.initialstates(rng, lkhood.Φ_fcns[i]) |> hq, lkhood.depth)
    layernorm_st = ()
    if lkhood.layernorm_bool && length(lkhood.layernorms) > 0
        layernorm_st = ntuple(i -> Lux.initialstates(rng, lkhood.layernorms[i]) |> hq, lkhood.depth-1)
    end

    batchnorm_st = ()
    if lkhood.batchnorm_bool && length(lkhood.batchnorms) > 0
        batchnorm_st = ntuple(i -> Lux.initialstates(rng, lkhood.batchnorms[i]) |> hq, lkhood.depth-1)
    end

    attention_st = ()
    if lkhood.seq_length > 1
        attention_st = (
            Q = Lux.initialstates(rng, lkhood.attention.Q) |> hq,
            K = Lux.initialstates(rng, lkhood.attention.K) |> hq,
            V = Lux.initialstates(rng, lkhood.attention.V) |> hq,
        )
    end

    return (
        fcn = fcn_st,
        layernorm = layernorm_st,
        batchnorm = batchnorm_st,
        attention = attention_st,
    )
end

end
