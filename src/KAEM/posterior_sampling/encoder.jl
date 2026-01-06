module EncoderModel

export EncoderWrapper, init_encoder

using Lux, ComponentArrays, Accessors, Random, ConfParser, NNlib

using ..Utils

struct BoolConfig <: AbstractBoolConfig
    variational::Bool
end

struct DiagonalGaussianEncoder <: Lux.AbstractLuxLayer
    depth::Int
    layers::Tuple{Vararg{Lux.Dense}}
    mu_head::Lux.Dense
    logvar_head::Lux.Dense
    latent_dim::Int
    input_dim::Int
end

struct NoEncoder <: Lux.AbstractLuxLayer end

struct EncoderWrapper{T <: Float32} <: Lux.AbstractLuxLayer
    encoder::Any
    bool_config::BoolConfig
    latent_dim::Int
    input_dim::Int
end

function init_diagonal_encoder(
        conf::ConfParse,
        x_shape::Tuple{Vararg{Int}},
        rng::AbstractRNG,
    )
    prior_widths = parse_config_array(Int, retrieve(conf, "EbmModel", "layer_widths"))
    q_size = first(prior_widths)
    p_size = last(prior_widths)
    latent_dim = q_size * p_size

    encoder_widths = parse_config_array(Int, retrieve(conf, "Encoder", "widths"))

    input_dim = prod(x_shape)
    layers = Vector{Lux.Dense}(undef, 0)

    prev_dim = input_dim
    for width in encoder_widths
        push!(layers, Lux.Dense(prev_dim => width, NNlib.gelu))
        prev_dim = width
    end

    mu_head = Lux.Dense(prev_dim => latent_dim)
    logvar_head = Lux.Dense(prev_dim => latent_dim)

    return DiagonalGaussianEncoder(
        length(layers),
        Tuple(layers),
        mu_head,
        logvar_head,
        latent_dim,
        input_dim,
    )
end

function init_no_encoder(
        conf::ConfParse,
        x_shape::Tuple{Vararg{Int}},
        rng::AbstractRNG,
    )
    return NoEncoder()
end

const encoder_map = Dict(
    "diagonal" => init_diagonal_encoder,
    "none" => init_no_encoder,
)

function init_encoder(
        conf::ConfParse,
        x_shape::Tuple{Vararg{Int}};
        rng::AbstractRNG = Random.default_rng(),
    )
    variational = parse(Bool, retrieve(conf, "VARIATIONAL", "use_variational"))
    encoder_type = variational ? "diagonal" : "none"

    prior_widths = parse_config_array(Int, retrieve(conf, "EbmModel", "layer_widths"))
    q_size = first(prior_widths)
    p_size = last(prior_widths)
    latent_dim = q_size * p_size
    input_dim = prod(x_shape)

    encoder_initializer = get(encoder_map, encoder_type, init_no_encoder)
    encoder = encoder_initializer(conf, x_shape, rng)

    return EncoderWrapper{Float32}(
        encoder,
        BoolConfig(variational),
        latent_dim,
        input_dim,
    )
end

function Lux.initialparameters(rng::AbstractRNG, wrapper::EncoderWrapper)
    if wrapper.bool_config.variational
        enc = wrapper.encoder
        layer_ps = NamedTuple(
            symbol_map[i] => Lux.initialparameters(rng, enc.layers[i])
                for i in 1:enc.depth
        )
        return (
            layers = layer_ps,
            mu = Lux.initialparameters(rng, enc.mu_head),
            logvar = Lux.initialparameters(rng, enc.logvar_head),
        )
    else
        return (a = [0.0f0], b = [0.0f0])
    end
end

function Lux.initialstates(rng::AbstractRNG, wrapper::EncoderWrapper)
    if wrapper.bool_config.variational
        enc = wrapper.encoder
        layer_st = NamedTuple(
            symbol_map[i] => Lux.initialstates(rng, enc.layers[i]) |> Lux.f32
                for i in 1:enc.depth
        )
        return (
            layers = layer_st,
            mu = Lux.initialstates(rng, enc.mu_head) |> Lux.f32,
            logvar = Lux.initialstates(rng, enc.logvar_head) |> Lux.f32,
        )
    else
        return (a = [0.0f0], b = [0.0f0])
    end
end

function (wrapper::EncoderWrapper)(
        ps,
        st_lux,
        x,
        ε,
    )
    if !wrapper.bool_config.variational
        return x, zeros(Float32, size(x)[end]), x, x, st_lux
    end

    enc = wrapper.encoder
    h = reshape(x, enc.input_dim, size(x)[end])
    st_new = st_lux

    for i in 1:enc.depth
        h, st_layer = Lux.apply(enc.layers[i], h, ps.layers[symbol_map[i]], st_new.layers[symbol_map[i]])
        @reset st_new.layers[symbol_map[i]] = st_layer
    end

    μ, st_mu = Lux.apply(enc.mu_head, h, ps.mu, st_new.mu)
    @reset st_new.mu = st_mu

    logvar, st_logvar = Lux.apply(enc.logvar_head, h, ps.logvar, st_new.logvar)
    @reset st_new.logvar = st_logvar

    logvar = clamp.(logvar, -10.0f0, 2.0f0)
    σ = exp.(0.5f0 .* logvar)
    z = μ .+ σ .* ε

    log_q = -0.5f0 .* sum(logvar .+ (z .- μ) .^ 2 ./ exp.(logvar) .+ log(2.0f0 * Float32(π)); dims = 1)
    log_q = dropdims(log_q; dims = 1)

    return z, log_q, μ, logvar, st_new
end

end
