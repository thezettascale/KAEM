module BaselineRNG

export seed_rng

using Random

using ..Utils
using ..VAEModel: VAE
using ..GANModel: GAN
using ..DDPMModel: DDPM
using ..PangEBMModel: PangEBM

function seed_rng(
        model::VAE{T};
        rng::AbstractRNG = Random.MersenneTwister(1),
    ) where {T <: Float32}
    return (
        ε = randn(rng, T, model.latent_dim, model.batch_size),
    ) |> pu
end

function seed_rng(
        model::GAN{T};
        rng::AbstractRNG = Random.MersenneTwister(1),
    ) where {T <: Float32}
    return (
        z = randn(rng, T, model.latent_dim, model.batch_size),
    ) |> pu
end

function seed_rng(
        model::DDPM{T};
        rng::AbstractRNG = Random.MersenneTwister(1),
        sampling::Bool = false,
    ) where {T <: Float32}

    if sampling
        return (
            x_init = randn(rng, T, model.x_shape..., model.batch_size),
            step_noise = randn(rng, T, model.x_shape..., model.batch_size, model.sampling_num_steps),
        ) |> pu
    end

    t_idx = rand(rng, 1:model.num_timesteps, model.batch_size)
    t_batch = T.(t_idx)
    broadcast_shape = (ones(Int, length(model.x_shape))..., model.batch_size)
    sqrt_alpha = reshape(model.sqrt_alphas_cumprod_vec[t_idx], broadcast_shape)
    sqrt_one_minus_alpha = reshape(model.sqrt_one_minus_alphas_cumprod_vec[t_idx], broadcast_shape)
    noise = randn(rng, T, model.x_shape..., model.batch_size)

    return (
        t = t_batch,
        sqrt_alpha = sqrt_alpha,
        sqrt_one_minus_alpha = sqrt_one_minus_alpha,
        noise = noise,
    ) |> pu
end

function seed_rng(
        model::PangEBM{T};
        rng::AbstractRNG = Random.MersenneTwister(1),
        batch_size::Int = model.batch_size,
    ) where {T <: Float32}

    latent_dim = model.latent_dim
    prior_steps = model.prior_sgld_steps
    post_steps = model.post_sgld_steps

    return (
        prior_init = randn(rng, T, latent_dim, batch_size),
        prior_noise = randn(rng, T, latent_dim, batch_size, prior_steps),
        post_init = randn(rng, T, latent_dim, batch_size),
        post_noise = randn(rng, T, latent_dim, batch_size, post_steps),
    ) |> pu
end

end
