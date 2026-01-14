module DDPMSampling

export sample_loop, seed_ddpm_rng

using Lux, Random
using Reactant: @trace

using ..Utils
using ..DDPMModel: DDPM

function denoise_step(
        x,
        t_float,
        alpha,
        alpha_cumprod,
        beta,
        noise,
        noise_mask,
        unet,
        ps,
        st
    )
    noise_pred, st_new = unet(x, t_float, ps, st)
    coef1 = 1.0f0 / sqrt(alpha)
    coef2 = beta / sqrt(1.0f0 - alpha_cumprod)
    mean = coef1 .* (x .- coef2 .* noise_pred)
    sigma = sqrt(beta)
    x_prev = mean .+ sigma .* noise .* noise_mask
    return x_prev, st_new
end

function seed_ddpm_rng(
        model::DDPM{T},
        x_shape,
        batch_size,
        stride::Int;
        rng::AbstractRNG = Random.MersenneTwister(1),
    ) where {T <: Float32}

    num_steps = cld(model.num_timesteps, stride)

    # Strided timestep (T, T-stride, T-2*stride, ..., down to 1)
    timesteps = Int[max(model.num_timesteps - (i - 1) * stride, 1) for i in 1:num_steps]

    alphas = Float32[model.alphas[t] for t in timesteps]
    alphas_cumprod = Float32[model.alphas_cumprod[t] for t in timesteps]
    betas = Float32[model.betas[t] for t in timesteps]
    t_floats = Float32.(timesteps)
    noise_masks = Float32[i < num_steps ? 1.0f0 : 0.0f0 for i in 1:num_steps]

    return (
        x_init = randn(rng, T, x_shape..., batch_size),
        step_noise = randn(rng, T, x_shape..., batch_size, num_steps),
        timesteps = t_floats,
        alphas = alphas,
        alphas_cumprod = alphas_cumprod,
        betas = betas,
        noise_masks = noise_masks,
        num_steps = num_steps,
    ) |> pu
end

function sample_loop(
        unet,
        ps,
        st,
        st_rng,
        batch_size::Int,
    )
    num_steps = st_rng.num_steps

    function step(i, x, st_curr)
        t_float_val = st_rng.timesteps[i]
        t_float = fill(t_float_val, batch_size)

        alpha = st_rng.alphas[i]
        alpha_cumprod = st_rng.alphas_cumprod[i]
        beta = st_rng.betas[i]
        noise_mask = st_rng.noise_masks[i]

        noise = selectdim(st_rng.step_noise, ndims(st_rng.step_noise), i)

        x_new, st_new = denoise_step(
            x, t_float, alpha, alpha_cumprod, beta, noise, noise_mask,
            unet, ps, st_curr
        )
        return x_new, st_new
    end

    x_init = st_rng.x_init
    state = (1, x_init, st)
    @trace while first(state) <= num_steps
        i, x_curr, st_curr = state
        x_new, st_new = step(i, x_curr, st_curr)
        state = (i + 1, x_new, st_new)
    end

    _, x_final, st_final = state
    x_clamped = clamp.(x_final, 0.0f0, 1.0f0)
    return x_clamped, st_final
end

end
