module HLOrng

using Random, ConfParser

export seed_rand

using ..Utils
using ..KAEM_model

function seed_rand(
        model::KAEM{T};
        rng::AbstractRNG = Random.MersenneTwister(1)
    ) where {T <: Float32}

    # ITS prior rng
    prior_its = (
        model.prior.bool_config.mixture_model ?
            rand(rng, T, model.prior.q_size, 1, 1, model.batch_size) :
            rand(rng, T, 1, model.prior.p_size, model.batch_size)
    )

    P = model.posterior_sampler.P
    num_temps = model.posterior_sampler.num_temps

    mix_mask_rv = (
        model.prior.bool_config.mixture_model ?
            rand(rng, T, model.prior.q_size, 1, model.batch_size) :
            [0.0f0]
    )

    mix_mask_rv_ula = (
        model.prior.bool_config.mixture_model ?
            rand(rng, T, model.prior.q_size, 1, model.batch_size * num_temps) :
            [0.0f0]
    )

    attn_rand = (
        model.prior.bool_config.use_attention_kernel ?
            rand(rng, T, model.prior.q_size, model.batch_size) :
            [0.0f0]
    )

    # ITS ula (init)
    posterior_its = (
        model.prior.bool_config.mixture_model ?
            rand(rng, T, model.prior.q_size, 1, 1, model.batch_size * num_temps) :
            rand(rng, T, 1, model.prior.p_size, model.batch_size * num_temps)
    )

    posterior_its = (
        num_temps > 1 || model.MALA ?
            posterior_its :
            (
                model.prior.prior_type == "uniform" ?
                rand(rng, T, P, 1, model.batch_size) :
                randn(rng, T, P, 1, model.batch_size)
            )
    )

    # Lkhood noise
    train_noise = (
        num_temps > 1 || model.MALA ?
            randn(rng, T, model.lkhood.x_shape..., model.batch_size) :
            randn(rng, T, model.lkhood.x_shape..., model.batch_size, model.batch_size)
    )

    tempered_noise = (
        num_temps > 1 ?
            randn(rng, T, model.lkhood.x_shape..., model.batch_size * (model.N_t + 1)) :
            [0.0f0]
    )

    # Resampler uniform noise
    systematic_bool = model.lkhood.resampler_type == "systematic"
    resample_rv = (
        (num_temps > 1 || model.MALA) ?
            [0.0f0] :
            (
                systematic_bool ?
                rand(rng, T, model.batch_size, 1, 1) :
                rand(rng, T, model.batch_size, model.batch_size, 1)
            )
    )

    Q, N, S = model.posterior_sampler.Q, model.posterior_sampler.N, model.batch_size
    ula_noise = randn(rng, T, Q, P, S * num_temps, N)
    log_swap = log.(rand(rng, T, S, num_temps, N))

    # Replica exchange masks and shift matrices
    exchange_type = (
        haskey(model.conf, "THERMODYNAMIC_INTEGRATION", "exchange_type") ?
            retrieve(model.conf, "THERMODYNAMIC_INTEGRATION", "exchange_type") :
            "deo"
    )

    if num_temps > 1 && exchange_type != "none"
        if exchange_type == "deo"
            even_1 = zeros(T, num_temps)
            even_2 = zeros(T, num_temps)
            for t in 1:2:(num_temps - 1)
                even_1[t] = 1.0f0
                even_2[t + 1] = 1.0f0
            end

            odd_1 = zeros(T, num_temps)
            odd_2 = zeros(T, num_temps)
            for t in 2:2:(num_temps - 1)
                odd_1[t] = 1.0f0
                odd_2[t + 1] = 1.0f0
            end

            swap_mask_1 = hcat([isodd(i) ? even_1 : odd_1 for i in 1:N]...)
            swap_mask_2 = hcat([isodd(i) ? even_2 : odd_2 for i in 1:N]...)
        else # random
            swap_mask_1 = zeros(T, num_temps, N)
            swap_mask_2 = zeros(T, num_temps, N)
            for i in 1:N
                t = rand(rng, 1:(num_temps - 1))
                swap_mask_1[t, i] = 1.0f0
                swap_mask_2[t + 1, i] = 1.0f0
            end
        end

        shift_down = zeros(T, num_temps, num_temps)
        shift_up = zeros(T, num_temps, num_temps)
        for t in 1:(num_temps - 1)
            shift_down[t, t + 1] = 1.0f0
            shift_up[t + 1, t] = 1.0f0
        end
    else
        swap_mask_1 = [0.0f0]
        swap_mask_2 = [0.0f0]
        shift_down = [0.0f0]
        shift_up = [0.0f0]
    end

    latent_dim = model.prior.q_size * model.prior.p_size
    encoder_noise = (
        model.encoder.bool_config.variational ?
            randn(rng, T, latent_dim, model.batch_size) :
            [0.0f0]
    )

    return (
        prior_its = prior_its,
        posterior_its = posterior_its,
        mix_rv = mix_mask_rv,
        mix_rv_ula = mix_mask_rv_ula,
        attn_rand = attn_rand,
        train_noise = train_noise,
        tempered_noise = tempered_noise,
        resample_rv = resample_rv,
        ula_noise = ula_noise,
        log_swap = log_swap,
        swap_mask_1 = swap_mask_1,
        swap_mask_2 = swap_mask_2,
        shift_down = shift_down,
        shift_up = shift_up,
        encoder_noise = encoder_noise,
    ) |> pu
end

end
