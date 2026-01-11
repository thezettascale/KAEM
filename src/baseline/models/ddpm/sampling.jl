module DDPMSampling

export q_sample, p_sample, sample_loop, seed_ddpm_sample_rng

using Lux, Random

using ..Utils
using ..DDPMModel: DDPM

function q_sample(x_0, sqrt_alpha, sqrt_one_minus_alpha, noise)
    return sqrt_alpha .* x_0 .+ sqrt_one_minus_alpha .* noise
end

function p_sample(
        model::DDPM,
        x_t,
        t_idx,
        t_float,
        alpha,
        alpha_cumprod,
        beta,
        ps,
        st,
        noise,
    )
    noise_pred, st_new = model(x_t, t_float, ps, st)

    coef1 = 1.0f0 ./ sqrt.(alpha)
    coef2 = beta ./ sqrt.(1.0f0 .- alpha_cumprod)
    mean = coef1 .* (x_t .- coef2 .* noise_pred)

    sigma = sqrt.(beta)
    x_prev = mean .+ sigma .* noise
    return x_prev, st_new
end

# Pre-generate all RNG (for Reactant compilation)
function seed_ddpm_sample_rng(
        model::DDPM{T},
        x_shape,
        batch_size;
        rng::AbstractRNG = Random.MersenneTwister(1),
    ) where {T <: Float32}
    num_t = model.num_timesteps

    x_init = randn(rng, T, x_shape..., batch_size)

    step_noise = cat(
        randn(rng, T, x_shape..., batch_size, num_t - 1),
        zeros(T, x_shape..., batch_size, 1);
        dims = 5
    )

    # Timestep floats (reverse order: T, T-1, ..., 1)
    t_floats = [fill(Float32(t), batch_size) for t in num_t:-1:1]
    t_floats = cat([reshape(tf, 1, :) for tf in t_floats]...; dims = 1)

    # Coefs for each timestep (reverse order)
    alphas = reshape(reverse(model.alphas), 1, 1, 1, 1, num_t)
    alphas_cumprod = reshape(reverse(model.alphas_cumprod), 1, 1, 1, 1, num_t)
    betas = reshape(reverse(model.betas), 1, 1, 1, 1, num_t)

    return (
        x_init = x_init |> pu,
        step_noise = step_noise |> pu,
        t_floats = t_floats |> pu,
        alphas = alphas |> pu,
        alphas_cumprod = alphas_cumprod |> pu,
        betas = betas |> pu,
    )
end

function sample_loop(model::DDPM, ps, st, st_rng)
    x = st_rng.x_init
    st_current = st
    num_t = model.num_timesteps

    for i in 1:num_t
        t_float = st_rng.t_floats[i, :]
        noise = st_rng.step_noise[:, :, :, :, i]
        alpha = st_rng.alphas[:, :, :, :, i]
        alpha_cumprod = st_rng.alphas_cumprod[:, :, :, :, i]
        beta = st_rng.betas[:, :, :, :, i]

        noise_pred, st_current = model(x, t_float, ps, st_current)

        coef1 = 1.0f0 ./ sqrt.(alpha)
        coef2 = beta ./ sqrt.(1.0f0 .- alpha_cumprod)
        mean = coef1 .* (x .- coef2 .* noise_pred)
        sigma = sqrt.(beta)
        x = mean .+ sigma .* noise
    end

    return clamp.(x, 0.0f0, 1.0f0), st_current
end

end
