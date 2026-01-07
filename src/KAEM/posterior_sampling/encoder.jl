module EncoderModel

export EncoderWrapper, init_encoder

using Lux, ComponentArrays, Accessors, Random, ConfParser

using ..Utils

include("models/diagonal.jl")
include("models/cnn.jl")
include("models/none.jl")
using .Diagonal_Model
using .CNN_Encoder_Model
using .NoEncoder_Model

struct BoolConfig <: AbstractBoolConfig
    variational::Bool
end

struct EncoderWrapper{T <: Float32} <: Lux.AbstractLuxLayer
    encoder::Any
    bool_config::BoolConfig
    latent_dim::Int
    input_dim::Int
    CNN::Bool
end

const encoder_map = Dict(
    "diagonal" => init_diagonal_encoder,
    "cnn" => init_cnn_encoder,
    "none" => init_no_encoder,
)

function init_encoder(
        conf::ConfParse,
        x_shape::Tuple{Vararg{Int}};
        rng::AbstractRNG = Random.default_rng(),
    )
    variational = parse(Bool, retrieve(conf, "VARIATIONAL", "use_variational"))
    encoder_type = variational ? retrieve(conf, "Encoder", "type") : "none"

    prior_widths = parse_config_array(Int, retrieve(conf, "EbmModel", "layer_widths"))
    q_size = first(prior_widths)
    p_size = last(prior_widths)
    latent_dim = q_size * p_size
    input_dim = prod(x_shape)

    encoder_initializer = get(encoder_map, encoder_type, init_diagonal_encoder)
    encoder = encoder_initializer(conf, x_shape, rng)

    return EncoderWrapper{Float32}(
        encoder,
        BoolConfig(variational),
        latent_dim,
        input_dim,
        encoder_type == "cnn",
    )
end

function Lux.initialparameters(rng::AbstractRNG, wrapper::EncoderWrapper)
    !wrapper.bool_config.variational && return EMPTY_PARAMS

    enc = wrapper.encoder

    if wrapper.CNN
        conv_ps = NamedTuple(
            symbol_map[i] => Lux.initialparameters(rng, enc.conv_layers[i])
                for i in 1:enc.depth
        )
        batchnorm_ps = init_optional_params(
            rng,
            enc.batchnorms,
            enc.bool_config.batchnorm
        )

        return (
            conv = conv_ps,
            batchnorm = batchnorm_ps,
            flatten = Lux.initialparameters(rng, enc.flatten_dense),
            mu = Lux.initialparameters(rng, enc.mu_head),
            logvar = Lux.initialparameters(rng, enc.logvar_head),
        )
    else
        layer_ps = NamedTuple(
            symbol_map[i] => Lux.initialparameters(rng, enc.layers[i])
                for i in 1:enc.depth
        )
        return (
            layers = layer_ps,
            mu = Lux.initialparameters(rng, enc.mu_head),
            logvar = Lux.initialparameters(rng, enc.logvar_head),
        )
    end
end

function Lux.initialstates(rng::AbstractRNG, wrapper::EncoderWrapper)
    !wrapper.bool_config.variational && return EMPTY_PARAMS

    enc = wrapper.encoder

    if wrapper.CNN
        conv_st = NamedTuple(
            symbol_map[i] => Lux.initialstates(rng, enc.conv_layers[i]) |> Lux.f32
                for i in 1:enc.depth
        )
        batchnorm_st = init_optional_states(
            rng,
            enc.batchnorms,
            enc.bool_config.batchnorm
        )

        return (
            conv = conv_st,
            batchnorm = batchnorm_st,
            flatten = Lux.initialstates(rng, enc.flatten_dense) |> Lux.f32,
            mu = Lux.initialstates(rng, enc.mu_head) |> Lux.f32,
            logvar = Lux.initialstates(rng, enc.logvar_head) |> Lux.f32,
        )
    else
        layer_st = NamedTuple(
            symbol_map[i] => Lux.initialstates(rng, enc.layers[i]) |> Lux.f32
                for i in 1:enc.depth
        )
        return (
            layers = layer_st,
            mu = Lux.initialstates(rng, enc.mu_head) |> Lux.f32,
            logvar = Lux.initialstates(rng, enc.logvar_head) |> Lux.f32,
        )
    end
end

function (wrapper::EncoderWrapper)(
        ps,
        st_lux,
        x,
        ε,
    )
    return wrapper.encoder(ps, st_lux, x, ε)
end

end
