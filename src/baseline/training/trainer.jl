module Baseline

export Trainer
export init_trainer, train!

using Lux, ComponentArrays, ConfParser, Random, Reactant, Optimisers
using Statistics, Flux, HDF5, JLD2, MLDataDevices
using Base: time

include("../../utils.jl")
using .Utils

include("../../pipeline/data_utils.jl")
using .DataUtils: get_vision_dataset

include("../../pipeline/optimizer.jl")
using .optimization

include("../models/ddpm/ddpm.jl")
include("../models/ddpm/sampling.jl")
include("../losses/ddpm_loss.jl")
using .DDPMModel
using .DDPMSampling
using .DDPMLoss

include("../models/gan/gan.jl")
include("../losses/gan_loss.jl")
using .GANModel
using .GANLoss

include("../models/pang_ebm/pang_ebm.jl")
include("../models/pang_ebm/sampling.jl")
include("../losses/pang_loss.jl")
using .PangEBMModel
using .PangEBMSampling
using .PangLoss

include("../models/vae/vae.jl")
include("../losses/vae_loss.jl")
using .VAEModel
using .VAELoss

include("setup.jl")
using .TrainingSetup

mutable struct Trainer{T <: Float32}
    model::Any
    train_step::Any
    gen_compiled::Any
    opt_state::Any
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

function init_trainer(
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
    checkpoint_every = -1
    gen_every = parse(Int, retrieve(conf, "TRAINING", "gen_every"))

    dataset, x_shape, save_dataset = get_vision_dataset(
        dataset_name, N_train, N_test, num_generated_samples;
        img_resize = img_resize, cnn = true,
    )

    train_data = dataset[:, :, :, 1:N_train]
    test_data = dataset[:, :, :, (N_train + 1):(N_train + N_test)]

    train_loader = Flux.DataLoader(train_data; batchsize = batch_size, shuffle = true)
    test_loader = Flux.DataLoader(test_data; batchsize = batch_size, shuffle = false)

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

    opt_state, opt_state_gen, opt_state_disc = nothing, nothing, nothing
    if model_type == :vae
        model = init_VAE(conf, x_shape; rng = rng)
        β = parse(Float32, retrieve(conf, "VAE", "beta"))
        model, train_step, opt_state, ps, st = prep_vae(
            model,
            x_sample,
            optimizer.rule();
            rng = rng,
            MLIR = MLIR,
            β = β
        )

        z_sample = randn(rng, Float32, model.latent_dim, batch_size) |> pu
        gen_compiled = if MLIR
            Reactant.@compile sample(
                model,
                ps,
                Lux.testmode(st),
                z_sample
            )
        else
            sample
        end

    elseif model_type == :gan
        model = init_GAN(conf, x_shape; rng = rng)
        n_critic = parse(Int, retrieve(conf, "GAN", "n_critic"))
        lr_gen = parse(Float32, retrieve(conf, "GAN", "lr_gen"))
        lr_disc = parse(Float32, retrieve(conf, "GAN", "lr_disc"))

        opt_gen = ManualAdam(lr_gen)
        opt_disc = ManualAdam(lr_disc)

        model, train_step, opt_state_gen, opt_state_disc, ps, st = prep_gan(
            model,
            x_sample,
            opt_gen,
            opt_disc;
            rng = rng,
            MLIR = MLIR,
            n_critic = n_critic
        )

        z_sample = randn(rng, Float32, model.latent_dim, batch_size) |> pu
        gen_compiled = if MLIR
            Reactant.@compile model.generator(
                z_sample,
                ps.gen,
                Lux.testmode(st.gen)
            )
        else
            generate
        end

    elseif model_type == :ddpm
        model = init_DDPM(conf, x_shape; rng = rng)
        model, train_step, opt_state, ps, st = prep_ddpm(
            model,
            x_sample,
            optimizer.rule();
            rng = rng,
            MLIR = MLIR
        )

        st_rng_sample = seed_ddpm_step_rng(model, x_shape, batch_size; rng = rng)
        gen_compiled = if MLIR
            Reactant.@compile denoise_step(
                model,
                st_rng_sample.x,
                st_rng_sample.t_float,
                st_rng_sample.alpha,
                st_rng_sample.alpha_cumprod,
                st_rng_sample.beta,
                st_rng_sample.noise,
                ps,
                Lux.testmode(st)
            )
        else
            denoise_step
        end

    elseif model_type == :pang
        model = init_PangEBM(conf, x_shape; rng = rng)
        α_cd = parse(Float32, retrieve(conf, "PANG", "alpha_cd"))
        model, train_step, opt_state, ps, st = prep_pang(
            model,
            x_sample,
            optimizer.rule();
            rng = rng,
            MLIR = MLIR,
            α_cd = α_cd
        )

        st_rng_sample = seed_pang_rng(model; rng = rng, batch_size = batch_size)
        gen_compiled = if MLIR
            Reactant.@compile generate_pang(
                model,
                ps,
                Lux.testmode(st),
                st_rng_sample
            )
        else
            generate_pang
        end
    else
        error("Unknown model type: $model_type. Use :vae, :gan, :ddpm, or :pang")
    end

    return Trainer{Float32}(
        model,
        train_step,
        gen_compiled,
        opt_state,
        opt_state_gen,
        opt_state_disc,
        ps,
        st,
        train_loader,
        test_loader,
        x_shape,
        batch_size,
        N_epochs,
        file_loc,
        num_generated_samples,
        checkpoint_every,
        gen_every,
        rng
    )

end

function save_checkpoint(t::Trainer, epoch::Int)
    return jldsave(
        t.file_loc * "ckpt_epoch_$(epoch).jld2";
        params = Array(t.ps),
        state = t.st |> MLDataDevices.cpu_device(),
    )
end

function log_loss(
        loss_file::String,
        now_time::Float64,
        epoch::Int,
        train_loss::Float32,
        test_loss::Float32 = 0.0f0
    )
    return open(loss_file, "a") do file
        write(file, "$now_time,$epoch,$train_loss,$test_loss\n")
    end
end

function prepare_batch(t::Trainer, x, train_idx = nothing)
    x = pu(x)
    if typeof(t.model) <: VAE
        ε = randn(t.rng, Float32, t.model.latent_dim, size(x, 4)) |> pu
        return (x, ε)
    elseif typeof(t.model) <: GAN
        z = randn(t.rng, Float32, t.model.latent_dim, size(x, 4)) |> pu
        return (x, z, train_idx)
    elseif typeof(t.model) <: DDPM
        batch_size = size(x, 4)
        t_idx = rand(t.rng, 1:t.model.num_timesteps, batch_size)
        t_batch = Float32.(t_idx) |> pu
        sqrt_alpha = t.model.sqrt_alphas_cumprod[t_idx]
        sqrt_one_minus_alpha = t.model.sqrt_one_minus_alphas_cumprod[t_idx]
        noise = randn(t.rng, Float32, t.x_shape..., batch_size) |> pu
        return (x, t_batch, sqrt_alpha, sqrt_one_minus_alpha, noise)
    elseif typeof(t.model) <: PangEBM
        batch_size = size(x, 4)
        st_rng = seed_pang_rng(t.model; rng = t.rng, batch_size = batch_size)
        return (x, st_rng)
    else
        error("Unknown model type: $(typeof(t.model))")
    end
end

function call_train_step(t::Trainer, batch_args)
    if typeof(t.model) <: VAE
        x, ε = batch_args
        return t.train_step(t.opt_state, t.ps, t.st, x, ε)
    elseif typeof(t.model) <: GAN
        x, z, train_idx = batch_args
        return t.train_step(
            t.opt_state_gen,
            t.opt_state_disc,
            t.ps,
            t.st,
            x,
            z,
            train_idx
        )
    elseif typeof(t.model) <: DDPM
        x, t_batch, sqrt_alpha, sqrt_one_minus_alpha, noise = batch_args
        return t.train_step(
            t.opt_state,
            t.ps,
            t.st,
            x,
            t_batch,
            sqrt_alpha,
            sqrt_one_minus_alpha,
            noise
        )
    elseif typeof(t.model) <: PangEBM
        x, st_rng = batch_args
        return t.train_step(t.opt_state, t.ps, t.st, x, st_rng)
    else
        error("Unknown model type: $(typeof(t.model))")
    end
end

image_test_loss(x, x_recon) = Flux.mse(x, x_recon)

function compute_test_loss(t::Trainer, test_step_compiled)
    if typeof(t.model) <: VAE
        test_loss = 0.0f0
        for x in t.test_loader
            x = pu(x)
            x_gen = generate_batch(t)
            test_loss += test_step_compiled(x, x_gen) |> Float32
            GC.gc()
        end
        return test_loss / length(t.test_loader)
    else
        return 0.0f0
    end
end

function generate_batch(t::Trainer)
    if typeof(t.model) <: VAE
        z = randn(t.rng, Float32, t.model.latent_dim, t.batch_size) |> pu
        return first(t.gen_compiled(t.model, t.ps, Lux.testmode(t.st), z))
    elseif typeof(t.model) <: GAN
        z = randn(t.rng, Float32, t.model.latent_dim, t.batch_size) |> pu
        return first(t.gen_compiled(z, t.ps.gen, Lux.testmode(t.st.gen)))
    elseif typeof(t.model) <: DDPM
        return first(
            sample_loop_eager(
                t.model,
                t.gen_compiled,
                t.ps,
                Lux.testmode(t.st),
                t.x_shape,
                t.batch_size;
                rng = t.rng
            )
        )
    elseif typeof(t.model) <: PangEBM
        st_rng = seed_pang_rng(t.model; rng = t.rng, batch_size = t.batch_size)
        return first(t.gen_compiled(t.model, t.ps, Lux.testmode(t.st), st_rng))
    else
        error("Unknown model type: $(typeof(t.model))")
    end
end

function save_generated_images(t::Trainer, gen_data, epoch; final::Bool = false)
    filename = final ? "generated_images.h5" : "generated_images_epoch_$(epoch).h5"
    return try
        h5write(t.file_loc * filename, "samples", gen_data)
    catch
        rm(t.file_loc * filename)
        h5write(t.file_loc * filename, "samples", gen_data)
    end
end

function train_loop!(
        t::Trainer;
        train_idx_start = 1,
        compute_test_loss = nothing,
    )
    num_batches = length(t.train_loader)
    loss_file = t.file_loc * "loss.csv"
    start_time = time()
    train_idx = train_idx_start

    for epoch in 1:t.N_epochs
        train_loss = 0.0f0

        for (batch_idx, x) in enumerate(t.train_loader)
            batch_args = prepare_batch(t, x, train_idx)
            result = call_train_step(t, batch_args)

            if !isnothing(t.opt_state_gen)
                loss, t.ps, t.opt_state_gen, t.opt_state_disc, t.st = result
            else
                loss, t.ps, t.opt_state, t.st = result
            end

            train_loss += Float32(loss)
            train_idx += 1
        end

        train_loss /= num_batches

        test_loss = isnothing(compute_test_loss) ? 0.0f0 : compute_test_loss(t)
        now_time = time() - start_time

        println(
            "Epoch: $epoch, Train Loss: $train_loss" *
                (test_loss > 0 ? ", Test Loss: $test_loss" : "")
        )
        log_loss(loss_file, now_time, epoch, train_loss, test_loss)

        if t.gen_every > 0 && epoch % t.gen_every == 0
            num_batches_gen = (t.num_generated_samples ÷ 10) ÷ t.batch_size
            if typeof(t.model) <: DDPM
                num_batches_gen = min(num_batches_gen, 10)
            end

            if num_batches_gen > 0
                first_batch = Array(generate_batch(t))
                batches_to_cat = Vector{typeof(first_batch)}()
                sizehint!(batches_to_cat, num_batches_gen)
                push!(batches_to_cat, first_batch)

                for _ in 2:num_batches_gen
                    push!(batches_to_cat, Array(generate_batch(t)))
                end

                gen_data = cat(batches_to_cat..., dims = 4)
                save_generated_images(t, gen_data, epoch)
            end
        end

        if t.checkpoint_every > 0 && epoch % t.checkpoint_every == 0
            save_checkpoint(t, epoch)
        end

        GC.gc()
    end

    num_batches_gen = t.num_generated_samples ÷ t.batch_size
    if typeof(t.model) <: DDPM
        num_batches_gen = min(num_batches_gen, 10)
    end

    return if num_batches_gen > 0
        first_batch = Array(generate_batch(t))
        batches_to_cat = Vector{typeof(first_batch)}()
        sizehint!(batches_to_cat, num_batches_gen)
        push!(batches_to_cat, first_batch)

        for _ in 2:num_batches_gen
            push!(batches_to_cat, Array(generate_batch(t)))
            GC.gc()
        end

        gen_data = cat(batches_to_cat..., dims = 4)
        save_generated_images(t, gen_data, t.N_epochs; final = true)
    end
end

function train!(t::Trainer)
    compute_test = nothing
    train_idx_start = 1

    if typeof(t.model) <: VAE
        x_sample = first(t.train_loader) |> pu
        x_gen_sample = generate_batch(t)
        test_step_compiled = Reactant.@compile image_test_loss(x_sample, x_gen_sample)

        compute_test = (t) -> compute_test_loss(t, test_step_compiled)
    elseif typeof(t.model) <: GAN
        train_idx_start = 1
    end

    return train_loop!(
        t;
        compute_test_loss = compute_test,
        train_idx_start = train_idx_start,
    )
end

end
