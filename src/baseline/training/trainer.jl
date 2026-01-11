module BaselineTrainer

export VAETrainer, GANTrainer, DDPMTrainer, PangEBMTrainer
export init_baseline_trainer, train!

using Lux, ComponentArrays, ConfParser, Random, Reactant, Optimisers
using Statistics, Flux, HDF5, JLD2, MLDataDevices
using Base: time

include("../../pipeline/data_utils.jl")
using .DataUtils: get_vision_dataset

include("../../pipeline/optimizer.jl")
using .optimization

using ..Utils
using ..VAEModel: VAE, init_VAE, sample
using ..GANModel: GAN, init_GAN
using ..DDPMModel: DDPM, init_DDPM
using ..PangEBMModel: PangEBM, init_PangEBM
using ..GANArchitecture: generate
using ..DDPMSampling: sample_loop, seed_ddpm_sample_rng
using ..PangEBMSampling: generate_pang, seed_pang_rng
using ..TrainingSetup: prep_vae, prep_gan, prep_ddpm, prep_pang

abstract type AbstractBaselineTrainer end

mutable struct VAETrainer{T <: Float32} <: AbstractBaselineTrainer
    model::VAE{T}
    train_step::Any
    gen_compiled::Any
    opt_state::Any
    ps::ComponentArray{T}
    st::NamedTuple
    train_loader::Any
    test_loader::Any
    x_shape::Tuple{Vararg{Int}}
    batch_size::Int
    N_epochs::Int
    file_loc::String
    num_generated_samples::Int
    checkpoint_every::Int
    gen_every::Int
    rng::AbstractRNG
end

mutable struct GANTrainer{T <: Float32} <: AbstractBaselineTrainer
    model::GAN{T}
    train_step::Any
    gen_compiled::Any
    opt_state_gen::Any
    opt_state_disc::Any
    ps::ComponentArray{T}
    st::NamedTuple
    train_loader::Any
    test_loader::Any
    x_shape::Tuple{Vararg{Int}}
    batch_size::Int
    N_epochs::Int
    file_loc::String
    num_generated_samples::Int
    checkpoint_every::Int
    gen_every::Int
    rng::AbstractRNG
end

mutable struct DDPMTrainer{T <: Float32} <: AbstractBaselineTrainer
    model::DDPM{T}
    train_step::Any
    gen_compiled::Any
    opt_state::Any
    ps::ComponentArray{T}
    st::NamedTuple
    train_loader::Any
    test_loader::Any
    x_shape::Tuple{Vararg{Int}}
    batch_size::Int
    N_epochs::Int
    file_loc::String
    num_generated_samples::Int
    checkpoint_every::Int
    gen_every::Int
    rng::AbstractRNG
end

mutable struct PangEBMTrainer{T <: Float32} <: AbstractBaselineTrainer
    model::PangEBM{T}
    train_step::Any
    gen_compiled::Any
    opt_state::Any
    ps::ComponentArray{T}
    st::NamedTuple
    train_loader::Any
    test_loader::Any
    x_shape::Tuple{Vararg{Int}}
    batch_size::Int
    N_epochs::Int
    file_loc::String
    num_generated_samples::Int
    checkpoint_every::Int
    gen_every::Int
    rng::AbstractRNG
end

function init_baseline_trainer(
        model_type::Symbol,
        conf::ConfParse,
        dataset_name::String;
        img_resize::Union{Nothing, Tuple{Int, Int}} = nothing,
        rng::AbstractRNG = Random.MersenneTwister(1),
        MLIR::Bool = true,
    )
    N_train = parse(Int, retrieve(conf, "TRAINING", "N_train"))
    N_test = parse(Int, retrieve(conf, "TRAINING", "N_test"))
    num_generated_samples = parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"))
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
    N_epochs = parse(Int, retrieve(conf, "TRAINING", "N_epochs"))
    checkpoint_every = -1  # Disabled to save storage
    gen_every = parse(Int, retrieve(conf, "TRAINING", "gen_every"))

    dataset, x_shape, save_dataset = get_vision_dataset(
        dataset_name, N_train, N_test, num_generated_samples;
        img_resize = img_resize, cnn = true,
    )

    train_data = dataset[:, :, :, 1:N_train]
    test_data = dataset[:, :, :, (N_train + 1):(N_train + N_test)]

    train_loader = [
        train_data[:, :, :, i:min(i + batch_size - 1, N_train)]
            for i in 1:batch_size:N_train
    ]
    test_loader = [
        test_data[:, :, :, i:min(i + batch_size - 1, N_test)]
            for i in 1:batch_size:N_test
    ]

    file_loc = "logs/Baseline/$(dataset_name)/$(uppercase(string(model_type)))/"
    mkpath(file_loc)

    try
        h5write(file_loc * "real_images.h5", "samples", save_dataset)
    catch
        rm(file_loc * "real_images.h5")
        h5write(file_loc * "real_images.h5", "samples", save_dataset)
    end

    open(file_loc * "loss.csv", "w") do file
        write(file, "Time (s),Epoch,Train Loss,Test Loss\n")
    end

    x_sample = first(train_loader) |> pu
    optimizer = create_opt(conf)

    if model_type == :vae
        model = init_VAE(conf, x_shape; rng = rng)
        β = parse(Float32, retrieve(conf, "VAE", "beta"))
        model, train_step, opt_state, ps, st = prep_vae(
            model, x_sample, optimizer.rule(); rng = rng, MLIR = MLIR, β = β
        )

        z_sample = randn(rng, Float32, model.latent_dim, batch_size) |> pu
        gen_compiled = if MLIR
            Reactant.@compile sample(model, ps, Lux.testmode(st), z_sample)
        else
            (m, p, s, z) -> sample(m, p, s, z)
        end

        return VAETrainer{Float32}(
            model, train_step, gen_compiled, opt_state, ps, st,
            train_loader, test_loader, x_shape, batch_size,
            N_epochs, file_loc, num_generated_samples,
            checkpoint_every, gen_every, rng
        )

    elseif model_type == :gan
        model = init_GAN(conf, x_shape; rng = rng)
        n_critic = parse(Int, retrieve(conf, "GAN", "n_critic"))
        lr_gen = parse(Float32, retrieve(conf, "GAN", "lr_gen"))
        lr_disc = parse(Float32, retrieve(conf, "GAN", "lr_disc"))

        opt_gen = ManualAdam(lr_gen)
        opt_disc = ManualAdam(lr_disc)

        model, train_step, opt_state_gen, opt_state_disc, ps, st = prep_gan(
            model, x_sample, opt_gen, opt_disc; rng = rng, MLIR = MLIR, n_critic = n_critic
        )

        z_sample = randn(rng, Float32, model.latent_dim, batch_size) |> pu
        gen_compiled = if MLIR
            Reactant.@compile generate(model.generator, z_sample, ps.gen, Lux.testmode(st.gen))
        else
            (gen, z, p, s) -> generate(gen, z, p, s)
        end

        return GANTrainer{Float32}(
            model, train_step, gen_compiled, opt_state_gen, opt_state_disc, ps, st,
            train_loader, test_loader, x_shape, batch_size,
            N_epochs, file_loc, num_generated_samples,
            checkpoint_every, gen_every, rng
        )

    elseif model_type == :ddpm
        model = init_DDPM(conf, x_shape; rng = rng)
        model, train_step, opt_state, ps, st = prep_ddpm(
            model, x_sample, optimizer.rule(); rng = rng, MLIR = MLIR
        )

        st_rng_sample = seed_ddpm_sample_rng(model, x_shape, batch_size; rng = rng)
        gen_compiled = if MLIR
            Reactant.@compile sample_loop(model, ps, Lux.testmode(st), st_rng_sample)
        else
            (m, p, s, sr) -> sample_loop(m, p, s, sr)
        end

        return DDPMTrainer{Float32}(
            model, train_step, gen_compiled, opt_state, ps, st,
            train_loader, test_loader, x_shape, batch_size,
            N_epochs, file_loc, num_generated_samples,
            checkpoint_every, gen_every, rng
        )

    elseif model_type == :pang
        model = init_PangEBM(conf, x_shape; rng = rng)
        α_cd = parse(Float32, retrieve(conf, "PANG", "alpha_cd"))
        model, train_step, opt_state, ps, st = prep_pang(
            model, x_sample, optimizer.rule(); rng = rng, MLIR = MLIR, α_cd = α_cd
        )

        st_rng_sample = seed_pang_rng(model; rng = rng, batch_size = batch_size)
        gen_compiled = if MLIR
            Reactant.@compile generate_pang(model, ps, Lux.testmode(st), st_rng_sample)
        else
            (m, p, s, sr) -> generate_pang(m, p, s, sr)
        end

        return PangEBMTrainer{Float32}(
            model, train_step, gen_compiled, opt_state, ps, st,
            train_loader, test_loader, x_shape, batch_size,
            N_epochs, file_loc, num_generated_samples,
            checkpoint_every, gen_every, rng
        )
    else
        error("Unknown model type: $model_type. Use :vae, :gan, :ddpm, or :pang")
    end
end

function train!(t::VAETrainer)
    num_batches = length(t.train_loader)
    loss_file = t.file_loc * "loss.csv"
    start_time = time()

    for epoch in 1:t.N_epochs
        train_loss = 0.0f0

        for (batch_idx, x) in enumerate(t.train_loader)
            x = pu(x)
            ε = randn(t.rng, Float32, t.model.latent_dim, size(x, 4)) |> pu

            loss, t.ps, t.opt_state, t.st = t.train_step(t.opt_state, t.ps, t.st, x, ε)
            train_loss += Float32(loss)
        end

        train_loss /= num_batches

        test_loss = 0.0f0
        for x in t.test_loader
            x = pu(x)
            ε = randn(t.rng, Float32, t.model.latent_dim, size(x, 4)) |> pu
            x_recon, _, _, _ = t.model(t.ps, Lux.testmode(t.st), x, ε)
            test_loss += Flux.mse(x, Array(x_recon))
        end
        test_loss /= length(t.test_loader)

        now_time = time() - start_time
        println("Epoch: $epoch, Train Loss: $train_loss, Test Loss: $test_loss")

        open(loss_file, "a") do file
            write(file, "$now_time,$epoch,$train_loss,$test_loss\n")
        end

        if t.gen_every > 0 && epoch % t.gen_every == 0
            generate_and_save_vae(t, epoch)
        end

        if t.checkpoint_every > 0 && epoch % t.checkpoint_every == 0
            jldsave(
                t.file_loc * "ckpt_epoch_$(epoch).jld2";
                params = Array(t.ps),
                state = t.st |> MLDataDevices.cpu_device(),
            )
        end

        GC.gc()
    end

    return generate_and_save_vae(t, t.N_epochs; final = true)
end

function train!(t::GANTrainer)
    num_batches = length(t.train_loader)
    loss_file = t.file_loc * "loss.csv"
    start_time = time()
    train_idx = 1

    for epoch in 1:t.N_epochs
        train_loss = 0.0f0

        for (batch_idx, x) in enumerate(t.train_loader)
            x = pu(x)
            z = randn(t.rng, Float32, t.model.latent_dim, size(x, 4)) |> pu

            loss, t.ps, t.opt_state_gen, t.opt_state_disc, t.st = t.train_step(
                t.opt_state_gen, t.opt_state_disc, t.ps, t.st, x, z, train_idx
            )
            train_loss += Float32(loss)
            train_idx += 1
        end

        train_loss /= num_batches

        now_time = time() - start_time
        println("Epoch: $epoch, Train Loss: $train_loss")

        open(loss_file, "a") do file
            write(file, "$now_time,$epoch,$train_loss,0.0\n")
        end

        if t.gen_every > 0 && epoch % t.gen_every == 0
            generate_and_save_gan(t, epoch)
        end

        if t.checkpoint_every > 0 && epoch % t.checkpoint_every == 0
            jldsave(
                t.file_loc * "ckpt_epoch_$(epoch).jld2";
                params = Array(t.ps),
                state = t.st |> MLDataDevices.cpu_device(),
            )
        end

        GC.gc()
    end

    return generate_and_save_gan(t, t.N_epochs; final = true)
end

function train!(t::DDPMTrainer)
    num_batches = length(t.train_loader)
    loss_file = t.file_loc * "loss.csv"
    start_time = time()

    for epoch in 1:t.N_epochs
        train_loss = 0.0f0

        for (batch_idx, x) in enumerate(t.train_loader)
            x = pu(x)
            batch_size = size(x, 4)
            t_idx = rand(t.rng, 1:t.model.num_timesteps, batch_size)
            t_batch = Float32.(t_idx) |> pu
            sqrt_alpha = reshape(t.model.sqrt_alphas_cumprod[t_idx], 1, 1, 1, :) |> pu
            sqrt_one_minus_alpha = reshape(t.model.sqrt_one_minus_alphas_cumprod[t_idx], 1, 1, 1, :) |> pu
            noise = randn(t.rng, Float32, t.x_shape..., batch_size) |> pu

            loss, t.ps, t.opt_state, t.st = t.train_step(
                t.opt_state, t.ps, t.st, x, t_batch, sqrt_alpha, sqrt_one_minus_alpha, noise
            )
            train_loss += Float32(loss)
        end

        train_loss /= num_batches

        now_time = time() - start_time
        println("Epoch: $epoch, Train Loss: $train_loss")

        open(loss_file, "a") do file
            write(file, "$now_time,$epoch,$train_loss,0.0\n")
        end

        if t.gen_every > 0 && epoch % t.gen_every == 0
            generate_and_save_ddpm(t, epoch)
        end

        if t.checkpoint_every > 0 && epoch % t.checkpoint_every == 0
            jldsave(
                t.file_loc * "ckpt_epoch_$(epoch).jld2";
                params = Array(t.ps),
                state = t.st |> MLDataDevices.cpu_device(),
            )
        end

        GC.gc()
    end

    return generate_and_save_ddpm(t, t.N_epochs; final = true)
end

function train!(t::PangEBMTrainer)
    num_batches = length(t.train_loader)
    loss_file = t.file_loc * "loss.csv"
    start_time = time()

    for epoch in 1:t.N_epochs
        train_loss = 0.0f0

        for (batch_idx, x) in enumerate(t.train_loader)
            x = pu(x)
            batch_size = size(x, 4)

            st_rng = seed_pang_rng(t.model; rng = t.rng, batch_size = batch_size)

            loss, t.ps, t.opt_state, t.st = t.train_step(t.opt_state, t.ps, t.st, x, st_rng)
            train_loss += Float32(loss)
        end

        train_loss /= num_batches

        now_time = time() - start_time
        println("Epoch: $epoch, Train Loss: $train_loss")

        open(loss_file, "a") do file
            write(file, "$now_time,$epoch,$train_loss,0.0\n")
        end

        if t.gen_every > 0 && epoch % t.gen_every == 0
            generate_and_save_pang(t, epoch)
        end

        if t.checkpoint_every > 0 && epoch % t.checkpoint_every == 0
            jldsave(
                t.file_loc * "ckpt_epoch_$(epoch).jld2";
                params = Array(t.ps),
                state = t.st |> MLDataDevices.cpu_device(),
            )
        end

        GC.gc()
    end

    return generate_and_save_pang(t, t.N_epochs; final = true)
end

function generate_and_save_vae(t::VAETrainer, epoch; final::Bool = false)
    num_batches = t.num_generated_samples ÷ t.batch_size
    batches = Vector{Array{Float32, 4}}()

    for _ in 1:num_batches
        z = randn(t.rng, Float32, t.model.latent_dim, t.batch_size) |> pu
        x_gen, _ = t.gen_compiled(t.model, t.ps, Lux.testmode(t.st), z)
        push!(batches, Array(x_gen))
    end

    gen_data = cat(batches..., dims = 4)
    filename = final ? "generated_images.h5" : "generated_images_epoch_$(epoch).h5"

    return try
        h5write(t.file_loc * filename, "samples", gen_data)
    catch
        rm(t.file_loc * filename)
        h5write(t.file_loc * filename, "samples", gen_data)
    end
end

function generate_and_save_gan(t::GANTrainer, epoch; final::Bool = false)
    num_batches = t.num_generated_samples ÷ t.batch_size
    batches = Vector{Array{Float32, 4}}()

    for _ in 1:num_batches
        z = randn(t.rng, Float32, t.model.latent_dim, t.batch_size) |> pu
        x_gen, _ = t.gen_compiled(t.model.generator, z, t.ps.gen, Lux.testmode(t.st.gen))
        push!(batches, Array(x_gen))
    end

    gen_data = cat(batches..., dims = 4)
    filename = final ? "generated_images.h5" : "generated_images_epoch_$(epoch).h5"

    return try
        h5write(t.file_loc * filename, "samples", gen_data)
    catch
        rm(t.file_loc * filename)
        h5write(t.file_loc * filename, "samples", gen_data)
    end
end

function generate_and_save_ddpm(t::DDPMTrainer, epoch; final::Bool = false)
    num_batches = min(t.num_generated_samples ÷ t.batch_size, 10)
    batches = Vector{Array{Float32, 4}}()

    for _ in 1:num_batches
        st_rng = seed_ddpm_sample_rng(t.model, t.x_shape, t.batch_size; rng = t.rng)
        x_gen, _ = t.gen_compiled(t.model, t.ps, Lux.testmode(t.st), st_rng)
        push!(batches, Array(x_gen))
    end

    gen_data = cat(batches..., dims = 4)
    filename = final ? "generated_images.h5" : "generated_images_epoch_$(epoch).h5"

    return try
        h5write(t.file_loc * filename, "samples", gen_data)
    catch
        rm(t.file_loc * filename)
        h5write(t.file_loc * filename, "samples", gen_data)
    end
end

function generate_and_save_pang(t::PangEBMTrainer, epoch; final::Bool = false)
    num_batches = t.num_generated_samples ÷ t.batch_size
    batches = Vector{Array{Float32, 4}}()

    for _ in 1:num_batches
        st_rng = seed_pang_rng(t.model; rng = t.rng, batch_size = t.batch_size)
        x_gen, _ = t.gen_compiled(t.model, t.ps, Lux.testmode(t.st), st_rng)
        push!(batches, Array(x_gen))
    end

    gen_data = cat(batches..., dims = 4)
    filename = final ? "generated_images.h5" : "generated_images_epoch_$(epoch).h5"

    return try
        h5write(t.file_loc * filename, "samples", gen_data)
    catch
        rm(t.file_loc * filename)
        h5write(t.file_loc * filename, "samples", gen_data)
    end
end

end
