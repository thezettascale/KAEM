module Baseline

export VAE, GAN, DDPM, PangEBM,
    init_VAE, init_GAN, init_DDPM, init_PangEBM,
    VAETrainer, GANTrainer, DDPMTrainer, PangEBMTrainer,
    init_baseline_trainer, train!,
    prep_vae, prep_gan, prep_ddpm, prep_pang,
    sample, generate, sample_loop, seed_ddpm_sample_rng,
    langevin_prior, langevin_posterior, generate_pang, seed_pang_rng,
    create_opt

using Lux, ComponentArrays, Accessors, Random, ConfParser, Reactant, Enzyme, Optimisers
using Statistics, LinearAlgebra, Flux, HDF5, JLD2, MLDataDevices

include("../utils.jl")
using .Utils

# Models - VAE
include("models/vae/architecture.jl")
include("models/vae/model.jl")

# Models - GAN
include("models/gan/architecture.jl")
include("models/gan/model.jl")

# Models - DDPM
include("models/ddpm/architecture.jl")
include("models/ddpm/model.jl")
include("models/ddpm/sampling.jl")

# Models - Pang EBM
include("models/pang_ebm/architecture.jl")
include("models/pang_ebm/model.jl")
include("models/pang_ebm/sampling.jl")

# Losses
include("losses/vae_loss.jl")
include("losses/gan_loss.jl")
include("losses/ddpm_loss.jl")
include("losses/pang_loss.jl")

# Training
include("training/setup.jl")
include("training/trainer.jl")

end
