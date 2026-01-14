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
    timesteps_idx = Int[max(model.num_timesteps - (i - 1) * stride, 1) for i in 1:num_steps]

    alphas_vec = Float32[model.alphas[t] for t in timesteps_idx]
    alphas_cumprod_vec = Float32[model.alphas_cumprod[t] for t in timesteps_idx]
    betas_vec = Float32[model.betas[t] for t in timesteps_idx]
    t_floats_vec = Float32.(timesteps_idx)
    noise_masks_vec = Float32[i < num_steps ? 1.0f0 : 0.0f0 for i in 1:num_steps]
    timesteps_batched = repeat(reshape(t_floats_vec, 1, num_steps), batch_size, 1)

    ndims_x = length(x_shape) + 1  # 4 dims for (H, W, C, batch)
    broadcast_shape = (ones(Int, ndims_x)..., num_steps)

    return (
        x_init = randn(rng, T, x_shape..., batch_size),
        step_noise = randn(rng, T, x_shape..., batch_size, num_steps),
        timesteps = timesteps_batched,
        alphas = reshape(alphas_vec, broadcast_shape...),
        alphas_cumprod = reshape(alphas_cumprod_vec, broadcast_shape...),
        betas = reshape(betas_vec, broadcast_shape...),
        noise_masks = reshape(noise_masks_vec, broadcast_shape...),
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
        mask = ((1:num_steps) .== i) |> Lux.f32
        t_mask = reshape(mask, 1, num_steps)
        t_float = dropdims(sum(st_rng.timesteps .* t_mask; dims = 2); dims = 2)

        sched_mask = reshape(mask, 1, 1, 1, 1, num_steps)
        alpha = dropdims(sum(st_rng.alphas .* sched_mask; dims = 5); dims = 5)
        alpha_cumprod = dropdims(sum(st_rng.alphas_cumprod .* sched_mask; dims = 5); dims = 5)
        beta = dropdims(sum(st_rng.betas .* sched_mask; dims = 5); dims = 5)
        noise_mask = dropdims(sum(st_rng.noise_masks .* sched_mask; dims = 5); dims = 5)
        noise_mask_5d = reshape(mask, 1, 1, 1, 1, num_steps)
        noise = dropdims(sum(st_rng.step_noise .* noise_mask_5d; dims = 5); dims = 5)

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
