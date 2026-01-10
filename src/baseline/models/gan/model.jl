export GAN, init_GAN

struct GAN{T <: Float32} <: Lux.AbstractLuxLayer
    generator::Generator
    discriminator::Discriminator
    latent_dim::Int
    x_shape::Tuple{Vararg{Int}}
    batch_size::Int
end

function init_GAN(
        conf::ConfParse,
        x_shape::Tuple{Vararg{Int}};
        rng::AbstractRNG = Random.default_rng(),
    )
    latent_dim = parse(Int, retrieve(conf, "GAN", "latent_dim"))
    gen_channels = parse.(Int, retrieve(conf, "GAN", "generator_channels"))
    disc_channels = parse.(Int, retrieve(conf, "GAN", "discriminator_channels"))
    kernel_size = parse(Int, retrieve(conf, "GAN", "kernel_size"))
    batchnorm = parse(Bool, retrieve(conf, "GAN", "batchnorm"))
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    generator = init_generator(x_shape, latent_dim, gen_channels, kernel_size, batchnorm)
    discriminator = init_discriminator(x_shape, disc_channels, kernel_size, batchnorm)

    return GAN{Float32}(generator, discriminator, latent_dim, x_shape, batch_size)
end

function Lux.initialparameters(rng::AbstractRNG, model::GAN)
    return (
        gen = Lux.initialparameters(rng, model.generator),
        disc = Lux.initialparameters(rng, model.discriminator),
    )
end

function Lux.initialstates(rng::AbstractRNG, model::GAN)
    return (
        gen = Lux.initialstates(rng, model.generator),
        disc = Lux.initialstates(rng, model.discriminator),
    )
end

function (model::GAN)(ps, st, z)
    x_gen, st_gen = generate(model.generator, z, ps.gen, st.gen)
    return x_gen, (gen = st_gen, disc = st.disc)
end
