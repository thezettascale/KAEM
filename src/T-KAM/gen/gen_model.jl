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
    Φ_fcns::Dict{Any,Any}
    layernorm::Bool
    batchnorm::Bool
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
    layernorm = parse(Bool, retrieve(conf, "GeneratorModel", "layer_norm"))
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
    batchnorm = false

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

    Φ_functions = Dict()
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

    if CNN
        channels = parse.(Int, retrieve(conf, "CNN", "hidden_feature_dims"))
        hidden_c = (q_size, channels...)
        depth = length(hidden_c)-1
        strides = parse.(Int, retrieve(conf, "CNN", "strides"))
        k_size = parse.(Int, retrieve(conf, "CNN", "kernel_sizes"))
        paddings = parse.(Int, retrieve(conf, "CNN", "paddings"))
        act = activation_mapping[retrieve(conf, "CNN", "activation")]
        batchnorm = parse(Bool, retrieve(conf, "CNN", "batchnorm"))
        generate_fcn = CNN_fwd
        layernorm = false

        length(strides) != length(hidden_c) &&
            (error("Number of strides must be equal to the number of hidden layers + 1."))
        length(k_size) != length(hidden_c) && (error(
            "Number of kernel sizes must be equal to the number of hidden layers + 1.",
        ))
        length(paddings) != length(hidden_c) &&
            (error("Number of paddings must be equal to the number of hidden layers + 1."))

        for i in eachindex(hidden_c[1:(end-1)])
            Φ_functions[Symbol("$i")] = Lux.ConvTranspose(
                (k_size[i], k_size[i]),
                hidden_c[i] => hidden_c[i+1],
                identity;
                stride = strides[i],
                pad = paddings[i],
            )
            if batchnorm
                Φ_functions[Symbol("bn_$i")] = Lux.BatchNorm(hidden_c[i+1], act)
            end
        end
        Φ_functions[Symbol("$(length(hidden_c))")] = Lux.ConvTranspose(
            (k_size[end], k_size[end]),
            hidden_c[end] => output_dim,
            identity;
            stride = strides[end],
            pad = paddings[end],
        )

    elseif sequence_length > 1

        act = gelu
        generate_fcn = SEQ_fwd

        # Single block Transformer decoder
        d_model = parse(Int, retrieve(conf, "SEQ", "d_model"))

        # Projection
        Φ_functions[Symbol("1")] = Lux.Dense(q_size => d_model)
        Φ_functions[Symbol("ln_1")] = Lux.LayerNorm((d_model, 1), gelu)

        # Query, Key, Value
        Φ_functions[Symbol("Q")] = Lux.Dense(d_model => d_model)
        Φ_functions[Symbol("K")] = Lux.Dense(d_model => d_model)
        Φ_functions[Symbol("V")] = Lux.Dense(d_model => d_model)

        # Feed forward
        Φ_functions[Symbol("2")] = Lux.Dense(d_model => d_model)
        Φ_functions[Symbol("ln_2")] = Lux.LayerNorm((d_model, 1), gelu)

        # Output layer
        Φ_functions[Symbol("3")] = Lux.Dense(d_model => output_dim)
        depth = 3
    else
        for i in eachindex(widths[1:(end-1)])
            base_scale = (
                μ_scale * (one(full_quant) / √(full_quant(widths[i]))) .+
                σ_base .* (
                    randn(rng, full_quant, widths[i], widths[i+1]) .* full_quant(2) .-
                    one(full_quant)
                ) .* (one(full_quant) / √(full_quant(widths[i])))
            )
            Φ_functions[Symbol("$i")] =
                initialize_function(widths[i], widths[i+1], base_scale)

            if (layernorm && i < depth)
                Φ_functions[Symbol("ln_$i")] = Lux.LayerNorm(widths[i+1])
            end
        end
    end

    return GenModel(
        Φ_functions,
        layernorm,
        batchnorm,
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

    ps = NamedTuple(
        Symbol("$i") => Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("$i")]) for
        i = 1:lkhood.depth
    )

    if lkhood.CNN
        @reset ps[Symbol("$(lkhood.depth+1)")] =
            Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("$(lkhood.depth+1)")])
        if lkhood.batchnorm
            for i = 1:lkhood.depth
                @reset ps[Symbol("bn_$i")] =
                    Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("bn_$i")])
            end
        end
    end

    if lkhood.layernorm
        for i = 1:(lkhood.depth-1)
            @reset ps[Symbol("ln_$i")] =
                Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("ln_$i")])
        end
    end

    if lkhood.seq_length > 1
        @reset ps[Symbol("Q")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("Q")])
        @reset ps[Symbol("K")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("K")])
        @reset ps[Symbol("V")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("V")])
    end

    return ps
end

function Lux.initialstates(rng::AbstractRNG, lkhood::GenModel{T}) where {T<:half_quant}

    st = NamedTuple(
        Symbol("$i") => Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("$i")]) |> hq for
        i = 1:lkhood.depth
    )

    if lkhood.CNN
        @reset st[Symbol("$(lkhood.depth+1)")] =
            Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("$(lkhood.depth+1)")]) |> hq

        if lkhood.batchnorm
            for i = 1:lkhood.depth
                @reset st[Symbol("bn_$i")] =
                    Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("bn_$i")]) |> hq
            end
        end
    end

    if lkhood.layernorm
        for i = 1:(lkhood.depth-1)
            @reset st[Symbol("ln_$i")] =
                Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("ln_$i")]) |> hq
        end
    end

    if lkhood.seq_length > 1
        @reset st[Symbol("Q")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("Q")])
        @reset st[Symbol("K")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("K")])
        @reset st[Symbol("V")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("V")])
    end

    return st
end

end
