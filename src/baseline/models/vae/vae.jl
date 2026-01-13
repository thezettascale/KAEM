module VAEModel

export VAE, init_VAE, sample, reparameterize

using Lux, ConfParser, Random, Accessors

using ..Utils

include("encoder.jl")
using .Encoder

include("decoder.jl")
using .Decoder

struct VAEConfig <: AbstractBoolConfig
    batchnorm::Bool
end

struct VAE{T <: Float32} <: Lux.AbstractLuxLayer
    encoder::VAEEncoder
    decoder::VAEDecoder
    latent_dim::Int
    x_shape::Tuple{Vararg{Int}}
    batch_size::Int
end

function init_VAE(
        conf::ConfParse,
        x_shape::Tuple{Vararg{Int}};
        rng::AbstractRNG = Random.default_rng(),
    )
    latent_dim = parse(Int, retrieve(conf, "VAE", "latent_dim"))
    enc_channels = parse.(Int, retrieve(conf, "VAE", "encoder_channels"))
    dec_channels = parse.(Int, retrieve(conf, "VAE", "decoder_channels"))
    enc_strides = parse.(Int, retrieve(conf, "VAE", "encoder_strides"))
    enc_kernels = parse.(Int, retrieve(conf, "VAE", "encoder_kernels"))
    enc_paddings = parse.(Int, retrieve(conf, "VAE", "encoder_paddings"))
    dec_strides = parse.(Int, retrieve(conf, "VAE", "decoder_strides"))
    dec_kernels = parse.(Int, retrieve(conf, "VAE", "decoder_kernels"))
    dec_paddings = parse.(Int, retrieve(conf, "VAE", "decoder_paddings"))
    batchnorm = parse(Bool, retrieve(conf, "VAE", "batchnorm"))
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    in_channels = last(x_shape)
    vae_conf = VAEConfig(batchnorm)

    encoder, spatial, channels = init_encoder(
        x_shape,
        enc_channels,
        latent_dim,
        enc_strides,
        enc_kernels,
        enc_paddings,
        vae_conf
    )

    decoder = init_decoder(
        in_channels,
        dec_channels,
        latent_dim,
        dec_strides,
        dec_kernels,
        dec_paddings,
        vae_conf,
        spatial,
        channels
    )

    return VAE{Float32}(
        encoder,
        decoder,
        latent_dim,
        x_shape,
        batch_size
    )
end

function Lux.initialparameters(rng::AbstractRNG, model::VAE)
    return (
        enc = Lux.initialparameters(rng, model.encoder),
        dec = Lux.initialparameters(rng, model.decoder),
    )
end

function Lux.initialstates(rng::AbstractRNG, model::VAE)
    return (
        enc = Lux.initialstates(rng, model.encoder),
        dec = Lux.initialstates(rng, model.decoder),
    )
end

function reparameterize(μ, logvar, ε)
    σ = exp.(0.5f0 .* logvar)
    return μ .+ σ .* ε
end

function (model::VAE)(ps, st, x, ε)
    μ, logvar, st_enc = model.encoder(x, ps.enc, st.enc)
    z = reparameterize(μ, logvar, ε)
    x_recon, st_dec = model.decoder(z, ps.dec, st.dec)
    @reset st.enc = st_enc
    @reset st.dec = st_dec
    return x_recon, μ, logvar, st
end

function sample(model::VAE, ps, st, z)
    x_gen, st_dec = model.decoder(z, ps.dec, st.dec)
    @reset st.dec = st_dec
    return x_gen, st
end

end
