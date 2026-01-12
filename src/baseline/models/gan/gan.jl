module GANModel

export GAN, init_GAN

using Lux, ConfParser, Random, Accessors

using ..Utils

include("discriminator.jl")
using .DiscriminatorGAN

include("generator.jl")
using .GeneratorGAN

struct GANConfig <: AbstractBoolConfig
    batchnorm::Bool
end

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
    gen_strides = parse.(Int, retrieve(conf, "GAN", "generator_strides"))
    gen_kernels = parse.(Int, retrieve(conf, "GAN", "generator_kernels"))
    gen_paddings = parse.(Int, retrieve(conf, "GAN", "generator_paddings"))
    disc_strides = parse.(Int, retrieve(conf, "GAN", "discriminator_strides"))
    disc_kernels = parse.(Int, retrieve(conf, "GAN", "discriminator_kernels"))
    disc_paddings = parse.(Int, retrieve(conf, "GAN", "discriminator_paddings"))
    batchnorm = parse(Bool, retrieve(conf, "GAN", "batchnorm"))
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    gan_conf = GANConfig(batchnorm)

    generator = init_generator(
        x_shape,
        latent_dim,
        gen_channels,
        gen_strides,
        gen_kernels,
        gen_paddings,
        gan_conf,
    )
    discriminator = init_discriminator(
        x_shape,
        disc_channels,
        disc_strides,
        disc_kernels,
        disc_paddings,
        gan_conf
    )

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
    x_gen, st_gen = model.generator(z, ps.gen, st.gen)
    @reset st.gen = st_gen
    return x_gen, st
end

end
