module GeneratorModel

export GenModel, init_GenModel, generator, importance_resampler

using ConfParser,
    Random, Lux, Statistics, LinearAlgebra, ComponentArrays, Accessors

using ..Utils
using ..UnivariateFunctions
using ..SymbolicFunctions

include("resamplers.jl")
include("models/kan.jl")
include("models/cnn.jl")
include("models/decoder.jl")
using .WeightResamplers
using .KAN_Model
using .CNN_Model
using .Transformer_Model

struct σ_conf{T <: Float32}
    noise::T
    llhood::T
end

const gen_model_map = Dict(
    "KAN" => init_KAN_Generator,
    "CNN" => init_CNN_Generator,
    "SEQ" => init_SEQ_Generator,
)

struct GenModel{T <: Float32} <: Lux.AbstractLuxLayer
    generator::Any
    σ::σ_conf{T}
    output_activation::AbstractActivation
    x_shape::Tuple{Vararg{Int}}
    resample_z::AbstractResampler
    resampler_type::AbstractString
    CNN::Bool
    SEQ::Bool
    perceptual_scale::T
end

function init_GenModel(
        conf::ConfParse,
        x_shape::Tuple{Vararg{Int}};
        rng::AbstractRNG = Random.default_rng(),
    )
    CNN = parse(Bool, retrieve(conf, "CNN", "use_cnn_lkhood"))
    sequence_length = parse(Int, retrieve(conf, "SEQ", "sequence_length"))

    noise_var = parse(Float32, retrieve(conf, "GeneratorModel", "generator_noise"))
    gen_var = parse(Float32, retrieve(conf, "GeneratorModel", "generator_variance"))
    ESS_threshold =
        parse(Float32, retrieve(conf, "TRAINING", "resampling_threshold_factor"))
    output_act = retrieve(conf, "GeneratorModel", "output_activation")
    verbose = parse(Bool, retrieve(conf, "TRAINING", "verbose"))

    resampler_type = retrieve(conf, "GeneratorModel", "resampler")
    resample_fcn = get(
        resampler_map,
        resampler_type,
        SystematicResampler(ESS_threshold, verbose)
    )(ESS_threshold, verbose)
    batchnorm_bool = false

    output_activation =
        sequence_length > 1 ? activation_mapping["sequence"] :
        get(activation_mapping, output_act, activation_mapping["identity"])

    gen_type = "KAN"

    if CNN
        gen_type = "CNN"
    elseif sequence_length > 1
        gen_type = "SEQ"
    end

    generator_initializer = get(gen_model_map, gen_type, init_KAN_Generator)
    generator = generator_initializer(conf, x_shape, rng)
    perceptual_scale =
        parse(Float32, retrieve(conf, "GeneratorModel", "perceptual_scale"))

    return GenModel(
        generator,
        σ_conf(noise_var, gen_var),
        output_activation,
        x_shape,
        resample_fcn,
        resampler_type,
        CNN,
        sequence_length > 1,
        perceptual_scale,
    )
end

function Lux.initialparameters(rng::AbstractRNG, lkhood::GenModel{T})::NamedTuple where {T <: Float32}
    fcn_ps = NamedTuple(
        symbol_map[i] => Lux.initialparameters(rng, lkhood.generator.Φ_fcns[i]) for
            i in 1:lkhood.generator.depth
    )
    layernorm_ps = (a = [0.0f0], b = [0.0f0])
    if lkhood.generator.bool_config.layernorm && length(lkhood.generator.layernorms) > 0
        layernorm_ps = NamedTuple(
            symbol_map[i] => Lux.initialparameters(rng, lkhood.generator.layernorms[i])
                for i in 1:length(lkhood.generator.layernorms)
        )
    end

    batchnorm_ps = (a = [0.0f0], b = [0.0f0])
    if lkhood.generator.bool_config.batchnorm && length(lkhood.generator.batchnorms) > 0
        batchnorm_ps = NamedTuple(
            symbol_map[i] => Lux.initialparameters(rng, lkhood.generator.batchnorms[i])
                for i in 1:length(lkhood.generator.batchnorms)
        )
    end

    attention_ps = (a = [0.0f0], b = [0.0f0])
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

function Lux.initialstates(rng::AbstractRNG, lkhood::GenModel{T})::Tuple{NamedTuple, NamedTuple} where {T <: Float32}
    fcn_st = NamedTuple(
        symbol_map[i] => Lux.initialstates(rng, lkhood.generator.Φ_fcns[i]) |> Lux.f32 for
            i in 1:lkhood.generator.depth
    )

    st_lyrnorm = (a = [0.0f0], b = [0.0f0])
    if lkhood.generator.bool_config.layernorm && length(lkhood.generator.layernorms) > 0
        st_lyrnorm = NamedTuple(
            symbol_map[i] =>
                Lux.initialstates(rng, lkhood.generator.layernorms[i]) |> Lux.f32 for
                i in 1:length(lkhood.generator.layernorms)
        )
    end

    batchnorm_st = (a = [0.0f0], b = [0.0f0])
    if lkhood.generator.bool_config.batchnorm && length(lkhood.generator.batchnorms) > 0
        batchnorm_st = NamedTuple(
            symbol_map[i] =>
                Lux.initialstates(rng, lkhood.generator.batchnorms[i]) |> Lux.f32 for
                i in 1:length(lkhood.generator.batchnorms)
        )
    end

    attention_st = (a = [0.0f0], b = [0.0f0])
    if lkhood.SEQ
        attention_st = (
            Q = Lux.initialstates(rng, lkhood.generator.attention[1]) |> Lux.f32,
            K = Lux.initialstates(rng, lkhood.generator.attention[2]) |> Lux.f32,
            V = Lux.initialstates(rng, lkhood.generator.attention[3]) |> Lux.f32,
        )
    end

    if lkhood.CNN || lkhood.SEQ
        return (a = [1.0f0], b = [1.0f0]),
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
