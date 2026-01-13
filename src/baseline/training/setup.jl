module TrainingSetup

export prep_vae, prep_gan, prep_ddpm, prep_pang

using Lux,
    ComponentArrays,
    Optimisers,
    Reactant,
    Random

using ..Utils
using ..VAEModel: VAE
using ..GANModel: GAN
using ..DDPMModel: DDPM
using ..PangEBMModel: PangEBM
using ..VAELoss: VAETrainStep
using ..GANLoss: GANTrainStep
using ..DDPMLoss: DDPMTrainStep
using ..PangLoss: PangTrainStep
using ..PangEBMSampling: seed_pang_rng

function prep_vae(
        model::VAE{T},
        x::AbstractArray{T},
        optimizer;
        rng::AbstractRNG = Random.MersenneTwister(1),
        MLIR::Bool = true,
        β::Float32 = 1.0f0,
    ) where {T <: Float32}

    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)
    ps, st = ps |> ComponentArray |> Lux.f32 |> pu, st |> Lux.f32 |> pu
    opt_state = Optimisers.setup(optimizer, ps)

    ε = randn(rng, T, model.latent_dim, model.batch_size) |> pu

    static_loss = VAETrainStep(model, β)

    compiled_step = if MLIR
        Reactant.@compile static_loss(opt_state, ps, st, x, ε)
    else
        static_loss
    end

    return model, compiled_step, opt_state, ps, st
end

function prep_gan(
        model::GAN{T},
        x::AbstractArray{T},
        optimizer_gen,
        optimizer_disc;
        rng::AbstractRNG = Random.MersenneTwister(1),
        MLIR::Bool = true,
        n_critic::Int = 1,
    ) where {T <: Float32}

    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)
    ps, st = ps |> ComponentArray |> Lux.f32 |> pu, st |> Lux.f32 |> pu

    opt_state_gen = Optimisers.setup(optimizer_gen, ps.gen)
    opt_state_disc = Optimisers.setup(optimizer_disc, ps.disc)

    z = randn(rng, T, model.latent_dim, model.batch_size) |> pu

    train_step = GANTrainStep(model, n_critic)

    compiled_step = if MLIR
        Reactant.@compile train_step(
            opt_state_gen, opt_state_disc, ps, st, x, z, 1
        )
    else
        train_step
    end

    return model, compiled_step, opt_state_gen, opt_state_disc, ps, st
end

function prep_ddpm(
        model::DDPM{T},
        x::AbstractArray{T},
        optimizer;
        rng::AbstractRNG = Random.MersenneTwister(1),
        MLIR::Bool = true,
    ) where {T <: Float32}

    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)
    ps, st = ps |> ComponentArray |> Lux.f32 |> pu, st |> Lux.f32 |> pu
    opt_state = Optimisers.setup(optimizer, ps)

    t_idx = rand(rng, 1:model.num_timesteps, model.batch_size)
    t = Float32.(t_idx) |> pu
    sqrt_alpha = model.sqrt_alphas_cumprod[ntuple(_ -> :, length(model.x_shape))..., t_idx] |> pu
    sqrt_one_minus_alpha = model.sqrt_one_minus_alphas_cumprod[ntuple(_ -> :, length(model.x_shape))..., t_idx] |> pu
    noise = randn(rng, T, model.x_shape..., model.batch_size) |> pu
    train_step = DDPMTrainStep(model)

    compiled_step = if MLIR
        Reactant.@compile train_step(
            opt_state, ps, st, x, t, sqrt_alpha, sqrt_one_minus_alpha, noise
        )
    else
        train_step
    end

    return model, compiled_step, opt_state, ps, st
end

function prep_pang(
        model::PangEBM{T},
        x::AbstractArray{T},
        optimizer;
        rng::AbstractRNG = Random.MersenneTwister(1),
        MLIR::Bool = true,
        α_cd::Float32 = 1.0f0,
    ) where {T <: Float32}

    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)
    ps, st = ps |> ComponentArray |> Lux.f32 |> pu, st |> Lux.f32 |> pu
    opt_state = Optimisers.setup(optimizer, ps)

    st_rng = seed_pang_rng(model; rng = rng, batch_size = model.batch_size)
    train_step = PangTrainStep(model, α_cd)

    compiled_step = if MLIR
        Reactant.@compile train_step(opt_state, ps, st, x, st_rng)
    else
        train_step
    end

    return model, compiled_step, opt_state, ps, st
end

end
