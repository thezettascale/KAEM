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
    coef1 = 1.0f0 ./ sqrt.(alpha)
    coef2 = beta ./ sqrt.(1.0f0 .- alpha_cumprod)
    mean = coef1 .* (x .- coef2 .* noise_pred)
    sigma = sqrt.(beta)
    x_prev = mean .+ sigma .* noise .* noise_mask
    return x_prev, st_new
end

function seed_ddpm_rng(
        model::DDPM{T};
        rng::AbstractRNG = Random.MersenneTwister(1),
    ) where {T <: Float32}
    return (
        x_init = randn(rng, T, model.x_shape..., model.batch_size),
        step_noise = randn(
            rng,
            T,
            model.x_shape...,
            model.batch_size,
            model.sampling_num_steps
        ),
    ) |> pu
end

function sample_loop(
        unet,
        ps,
        st,
        st_rng,
        timesteps,
        alphas,
        alphas_cumprod,
        betas,
        noise_masks,
        num_steps::Int,
    )
    function step(i, x, st_curr)
        mask = ((1:num_steps) .== i) |> Lux.f32
        t_mask = reshape(mask, 1, num_steps)
        t_float = dropdims(sum(timesteps .* t_mask; dims = 2); dims = 2)

        sched_mask = reshape(mask, 1, 1, 1, 1, num_steps)
        alpha = dropdims(sum(alphas .* sched_mask; dims = 5); dims = 5)
        alpha_cumprod_i = dropdims(sum(alphas_cumprod .* sched_mask; dims = 5); dims = 5)
        beta = dropdims(sum(betas .* sched_mask; dims = 5); dims = 5)
        noise_mask = dropdims(sum(noise_masks .* sched_mask; dims = 5); dims = 5)

        noise_mask_5d = reshape(mask, 1, 1, 1, 1, num_steps)
        noise = dropdims(sum(st_rng.step_noise .* noise_mask_5d; dims = 5); dims = 5)

        x_new, st_new = denoise_step(
            x, t_float, alpha, alpha_cumprod_i, beta, noise, noise_mask,
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
