module GeneratorModel

export GenModel,
    init_GenModel, generator, importance_resampler, log_likelihood_IS, log_likelihood_MALA

using CUDA, KernelAbstractions
using ConfParser,
    Random, Lux, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays, Accessors
using NNlib: sigmoid_fast, tanh_fast, relu, gelu, sigmoid, tanh

include("../ebm/ebm_model.jl")
include("../../utils.jl")
include("resamplers.jl")
include("loglikelihoods.jl")
include("models/kan.jl")
include("models/cnn.jl")
include("models/decoder.jl")
using .Utils: device, half_quant, full_quant, hq, fq, symbol_map
using .EBM_Model
using .WeightResamplers
using .LogLikelihoods
using .KAN_Model
using .CNN_Model
using .Transformer_Model

const output_activation_mapping =
    Dict("tanh" => tanh_fast, "sigmoid" => sigmoid_fast, "none" => identity)

const resampler_map = Dict(
    "residual" => residual_resampler,
    "systematic" => systematic_resampler,
    "stratified" => stratified_resampler,
)

const gen_model_map = Dict(
    "KAN" => init_KAN_Generator,
    "CNN" => init_CNN_Generator,
    "SEQ" => init_SEQ_Generator,
)

struct GenModel{T<:half_quant} <: Lux.AbstractLuxLayer
    generator::Any
    σ_llhood::T
    output_activation::Function
    x_shape::Tuple{Vararg{Int}}
    resample_z::Function
    CNN::Bool
    SEQ::Bool
end

function init_GenModel(
    conf::ConfParse,
    x_shape::Tuple{Vararg{Int}};
    rng::AbstractRNG = Random.default_rng(),
)
    CNN = parse(Bool, retrieve(conf, "CNN", "use_cnn_lkhood"))
    sequence_length = parse(Int, retrieve(conf, "SEQ", "sequence_length"))

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

    output_activation =
        sequence_length > 1 ? (x -> softmax(x, dims = 1)) :
        get(output_activation_mapping, output_act, identity)

    gen_type = "KAN"

    if CNN
        gen_type = "CNN"
    elseif sequence_length > 1
        gen_type = "SEQ"
    end

    generator_initializer = get(gen_model_map, gen_type, init_KAN_Generator)
    generator = generator_initializer(conf, x_shape, rng)

    return GenModel(
        generator,
        gen_var,
        output_activation,
        x_shape,
        resample_fcn,
        CNN,
        sequence_length > 1,
    )
end

function Lux.initialparameters(rng::AbstractRNG, lkhood::GenModel{T}) where {T<:half_quant}
    fcn_ps = NamedTuple(
        symbol_map[i] => Lux.initialparameters(rng, lkhood.generator.Φ_fcns[i]) for
        i in eachindex(lkhood.generator.Φ_fcns)
    )
    layernorm_ps = (a = zero(T))
    if lkhood.generator.layernorm_bool && length(lkhood.generator.layernorms) > 0
        layernorm_ps = NamedTuple(
            symbol_map[i] => Lux.initialparameters(rng, lkhood.generator.layernorms[i])
            for i in eachindex(lkhood.generator.layernorms)
        )
    end

    batchnorm_ps = (a = zero(T))
    if lkhood.generator.batchnorm_bool && length(lkhood.generator.batchnorms) > 0
        batchnorm_ps = NamedTuple(
            symbol_map[i] => Lux.initialparameters(rng, lkhood.generator.batchnorms[i])
            for i in eachindex(lkhood.generator.batchnorms)
        )
    end

    attention_ps = (a = zero(T))
    if lkhood.SEQ
        attention_ps = (
            Q = Lux.initialparameters(rng, lkhood.generator.attention[1]),
            K = Lux.initialparameters(rng, lkhood.generator.attention[2]),
            V = Lux.initialparameters(rng, lkhood.generator.attention[3]),
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
    fcn_st = NamedTuple(
        symbol_map[i] => Lux.initialstates(rng, lkhood.generator.Φ_fcns[i]) |> hq for
        i in eachindex(lkhood.generator.Φ_fcns)
    )

    st_lyrnorm = (a = zero(T), b = zero(T))
    if lkhood.generator.layernorm_bool && length(lkhood.generator.layernorms) > 0
        st_lyrnorm = NamedTuple(
            symbol_map[i] =>
                Lux.initialstates(rng, lkhood.generator.layernorms[i]) |> hq for
            i in eachindex(lkhood.generator.layernorms)
        )
    end

    batchnorm_st = (a = zero(T))
    if lkhood.generator.batchnorm_bool && length(lkhood.generator.batchnorms) > 0
        batchnorm_st = NamedTuple(
            symbol_map[i] =>
                Lux.initialstates(rng, lkhood.generator.batchnorms[i]) |> hq for
            i in eachindex(lkhood.generator.batchnorms)
        )
    end

    attention_st = (a = zero(T))
    if lkhood.SEQ
        attention_st = (
            Q = Lux.initialstates(rng, lkhood.generator.attention[1]) |> hq,
            K = Lux.initialstates(rng, lkhood.generator.attention[2]) |> hq,
            V = Lux.initialstates(rng, lkhood.generator.attention[3]) |> hq,
        )
    end

    if lkhood.CNN || lkhood.SEQ
        return (a = one(T), b = one(T)),
        (
            fcn = fcn_st,
            layernorm = st_lyrnorm,
            batchnorm = batchnorm_st,
            attention = attention_st,
        )
    else
        return fcn_st, st_lyrnorm
    end
end

end
