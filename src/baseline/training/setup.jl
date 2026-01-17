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
using ..BaselineRNG: seed_rng

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

    st_rng = seed_rng(model; rng = rng)

    static_loss = VAETrainStep(model, β)

    println("  Compiling VAE train step...")
    compiled_step = if MLIR
        Reactant.@compile static_loss(opt_state, ps, Lux.trainmode(st), x, st_rng)
    else
        static_loss
    end
    println("  VAE train step compiled.")

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

    st_rng = seed_rng(model; rng = rng)

    train_step = GANTrainStep(model, n_critic)

    println("  Compiling GAN train step...")
    compiled_step = if MLIR
        Reactant.@compile train_step(
            opt_state_gen, opt_state_disc, ps, Lux.trainmode(st), x, st_rng, 1
        )
    else
        train_step
    end
    println("  GAN train step compiled.")

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

    st_rng = seed_rng(model; rng = rng)
    train_step = DDPMTrainStep(model)

    println("  Compiling DDPM train step...")
    # Pass trainmode state to Reactant - it needs to compile with training mode
    # so any normalization layers use training mode (which has gradient support)
    st_train = Lux.trainmode(st)
    compiled_step = if MLIR
        Reactant.@compile train_step(opt_state, ps, st_train, x, st_rng)
    else
        train_step
    end
    println("  DDPM train step compiled.")

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

    st_rng = seed_rng(model; rng = rng, batch_size = model.batch_size)
    train_step = PangTrainStep(model, α_cd)

    println("  Compiling Pang train step...")
    compiled_step = if MLIR
        Reactant.@compile train_step(opt_state, ps, Lux.trainmode(st), x, st_rng)
    else
        train_step
    end
    println("  Pang train step compiled.")

    return model, compiled_step, opt_state, ps, st
end

end
