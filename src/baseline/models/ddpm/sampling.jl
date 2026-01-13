module DDPMSampling

export denoise_step, sample_loop_eager, seed_ddpm_step_rng

using Lux, Random

using ..Utils
using ..DDPMModel: DDPM

function q_sample(x_0, sqrt_alpha, sqrt_one_minus_alpha, noise)
    return sqrt_alpha .* x_0 .+ sqrt_one_minus_alpha .* noise
end

function denoise_step(
        model,
        x,
        t_float,
        alpha,
        alpha_cumprod,
        beta,
        noise,
        ps,
        st
    )
    noise_pred, st_new = model(x, t_float, ps, st)
    coef1 = 1.0f0 ./ sqrt.(alpha)
    coef2 = beta ./ sqrt.(1.0f0 .- alpha_cumprod)
    mean = coef1 .* (x .- coef2 .* noise_pred)
    sigma = sqrt.(beta)
    x_prev = mean .+ sigma .* noise
    return x_prev, st_new
end

# Seed RNG for a single step (for compilation tracing)
function seed_ddpm_step_rng(
        model::DDPM{T},
        x_shape,
        batch_size;
        rng::AbstractRNG = Random.MersenneTwister(1),
    ) where {T <: Float32}
    x_sample = randn(rng, T, x_shape..., batch_size) |> pu
    t_float = fill(Float32(model.num_timesteps), batch_size) |> pu
    noise = randn(rng, T, x_shape..., batch_size) |> pu

    alpha = model.alphas[1]
    alpha_cumprod = model.alphas_cumprod[1]
    beta = model.betas[1]

    return (
        x = x_sample,
        t_float = t_float,
        alpha = alpha,
        alpha_cumprod = alpha_cumprod,
        beta = beta,
        noise = noise,
    )
end

function sample_loop_eager(
        model::DDPM{T},
        step_compiled,
        ps,
        st,
        x_shape,
        batch_size;
        rng::AbstractRNG = Random.MersenneTwister(1),
    ) where {T <: Float32}
    num_t = model.num_timesteps

    x = randn(rng, T, x_shape..., batch_size) |> pu
    st_current = st

    # Denoising: t = T, T-1, ..., 1
    for t_idx in num_t:-1:1
        t_float = fill(Float32(t_idx), batch_size) |> pu
        alpha = model.alphas[t_idx]
        alpha_cumprod = model.alphas_cumprod[t_idx]
        beta = model.betas[t_idx]

        noise = if t_idx > 1
            randn(rng, T, x_shape..., batch_size) |> pu
        else
            zeros(T, x_shape..., batch_size) |> pu
        end

        x, st_current = step_compiled(
            model, x, t_float, alpha, alpha_cumprod, beta, noise, ps, st_current
        )

        # Force GC
        if t_idx % 100 == 0
            GC.gc()
        end
    end

    return clamp.(x, 0.0f0, 1.0f0), st_current
end

end
