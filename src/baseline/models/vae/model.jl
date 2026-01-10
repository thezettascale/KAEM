export VAE, init_VAE, sample, reparameterize

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
    kernel_size = parse(Int, retrieve(conf, "VAE", "kernel_size"))
    batchnorm = parse(Bool, retrieve(conf, "VAE", "batchnorm"))
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    in_channels = last(x_shape)

    encoder, spatial, channels = init_encoder(
        x_shape, enc_channels, latent_dim, kernel_size, batchnorm
    )

    decoder = init_decoder(
        in_channels, dec_channels, latent_dim, kernel_size, batchnorm, spatial, channels
    )

    return VAE{Float32}(encoder, decoder, latent_dim, x_shape, batch_size)
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
    μ, logvar, st_enc = encode(model.encoder, x, ps.enc, st.enc)
    z = reparameterize(μ, logvar, ε)
    x_recon, st_dec = decode(model.decoder, z, ps.dec, st.dec)
    return x_recon, μ, logvar, (enc = st_enc, dec = st_dec)
end

function sample(model::VAE, ps, st, z)
    x_gen, st_dec = decode(model.decoder, z, ps.dec, st.dec)
    return x_gen, st_dec
end
