module Baseline

export Trainer
export init_trainer, train!

using Lux, ComponentArrays, ConfParser, Random, Reactant, Optimisers
using Statistics, Flux, HDF5, JLD2, Accessors
using MLDataDevices: cpu_device
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

function prepare_batch_vae(model, rng, x_shape, x, train_idx)
    ε = randn(rng, Float32, model.latent_dim, model.batch_size) |> pu
    return (x, ε)
end

function prepare_batch_gan(model, rng, x_shape, x, train_idx)
    z = randn(rng, Float32, model.latent_dim, model.batch_size) |> pu
    return (x, z, train_idx)
end

function prepare_batch_ddpm(model, rng, x_shape, x, train_idx)
    t_idx = rand(rng, 1:model.num_timesteps, model.batch_size)
    t_batch = Float32.(t_idx) |> pu
    sqrt_alpha_gathered = model.sqrt_alphas_cumprod_vec[t_idx]
    sqrt_one_minus_alpha_gathered = model.sqrt_one_minus_alphas_cumprod_vec[t_idx]
    broadcast_shape = (ones(Int, length(x_shape))..., model.batch_size)
    sqrt_alpha = collect(reshape(sqrt_alpha_gathered, broadcast_shape)) |> pu
    sqrt_one_minus_alpha = collect(reshape(sqrt_one_minus_alpha_gathered, broadcast_shape)) |> pu
    noise = randn(rng, Float32, x_shape..., model.batch_size) |> pu
    return (x, t_batch, sqrt_alpha, sqrt_one_minus_alpha, noise)
end

function prepare_batch_pang(model, rng, x_shape, x, train_idx)
    st_rng = seed_pang_rng(model; rng = rng, batch_size = model.batch_size)
    return (x, st_rng)
end

function call_train_step_vae(
        train_step,
        opt_state,
        opt_state_gen,
        opt_state_disc,
        ps,
        st,
        batch_args
    )
    x, ε = batch_args
    loss, ps, opt_state, st = train_step(opt_state, ps, st, x, ε)
    return (loss, ps, opt_state, opt_state, opt_state, st)
end

function call_train_step_gan(
        train_step,
        opt_state,
        opt_state_gen,
        opt_state_disc,
        ps,
        st,
        batch_args
    )
    x, z, train_idx = batch_args
    loss, ps, opt_state_gen, opt_state_disc, st = train_step(
        opt_state_gen,
        opt_state_disc,
        ps,
        st,
        x,
        z,
        train_idx
    )
    return (
        loss,
        ps,
        opt_state,
        opt_state_gen,
        opt_state_disc,
        st,
    )
end

function call_train_step_ddpm(
        train_step,
        opt_state,
        opt_state_gen,
        opt_state_disc,
        ps,
        st,
        batch_args
    )
    x, t_batch, sqrt_alpha, sqrt_one_minus_alpha, noise = batch_args
    loss, ps, opt_state, st = train_step(
        opt_state,
        ps,
        st,
        x,
        t_batch,
        sqrt_alpha,
        sqrt_one_minus_alpha,
        noise
    )
    return (
        loss,
        ps,
        opt_state,
        opt_state,
        opt_state,
        st,
    )
end

function call_train_step_pang(
        train_step,
        opt_state,
        opt_state_gen,
        opt_state_disc,
        ps,
        st,
        batch_args
    )
    x, st_rng = batch_args
    loss, ps, opt_state, st = train_step(opt_state, ps, st, x, st_rng)
    return (loss, ps, opt_state, opt_state, opt_state, st)
end

function generate_batch_vae(
        model,
        gen_compiled,
        ps,
        st,
        rng,
        x_shape,
        batch_size
    )
    z = randn(rng, Float32, model.latent_dim, model.batch_size) |> pu
    x_gen, st_new = gen_compiled(model, ps, Lux.testmode(st), z)
    return x_gen, st_new
end

function generate_batch_gan(
        model,
        gen_compiled,
        ps,
        st,
        rng,
        x_shape,
        batch_size
    )
    z = randn(rng, Float32, model.latent_dim, model.batch_size) |> pu
    x_gen, st_gen_new = gen_compiled(z, ps.gen, Lux.testmode(st.gen))
    @reset st.gen = st_gen_new
    return x_gen, st
end

function generate_batch_ddpm(
        model,
        gen_compiled,
        ps,
        st,
        rng,
        x_shape,
        batch_size
    )
    x_gen, st_new = sample_loop_eager(
        model,
        gen_compiled,
        ps,
        Lux.testmode(st),
        x_shape,
        model.batch_size;
        rng = rng
    )
    return x_gen, st_new
end

function generate_batch_pang(
        model,
        gen_compiled,
        ps,
        st,
        rng,
        x_shape,
        batch_size
    )
    st_rng = seed_pang_rng(model; rng = rng, batch_size = model.batch_size)
    x_gen, st_new = gen_compiled(model, ps, Lux.testmode(st), st_rng)
    return x_gen, st_new
end

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
    prepare_batch_fn::Any
    call_train_step_fn::Any
    generate_batch_fn::Any
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

    prepare_batch_fns = Dict(
        :vae => (x, train_idx, rng, x_shape) -> prepare_batch_vae(model, rng, x_shape, x, train_idx),
        :gan => (x, train_idx, rng, x_shape) -> prepare_batch_gan(model, rng, x_shape, x, train_idx),
        :ddpm => (x, train_idx, rng, x_shape) -> prepare_batch_ddpm(model, rng, x_shape, x, train_idx),
        :pang => (x, train_idx, rng, x_shape) -> prepare_batch_pang(model, rng, x_shape, x, train_idx),
    )

    call_train_step_fns = Dict(
        :vae => (train_step, opt_state, opt_state_gen, opt_state_disc, ps, st, batch_args) -> call_train_step_vae(train_step, opt_state, opt_state_gen, opt_state_disc, ps, st, batch_args),
        :gan => (train_step, opt_state, opt_state_gen, opt_state_disc, ps, st, batch_args) -> call_train_step_gan(train_step, opt_state, opt_state_gen, opt_state_disc, ps, st, batch_args),
        :ddpm => (train_step, opt_state, opt_state_gen, opt_state_disc, ps, st, batch_args) -> call_train_step_ddpm(train_step, opt_state, opt_state_gen, opt_state_disc, ps, st, batch_args),
        :pang => (train_step, opt_state, opt_state_gen, opt_state_disc, ps, st, batch_args) -> call_train_step_pang(train_step, opt_state, opt_state_gen, opt_state_disc, ps, st, batch_args),
    )

    generate_batch_fns = Dict(
        :vae => (gen_compiled, ps, st, rng, x_shape, batch_size) -> generate_batch_vae(model, gen_compiled, ps, st, rng, x_shape, batch_size),
        :gan => (gen_compiled, ps, st, rng, x_shape, batch_size) -> generate_batch_gan(model, gen_compiled, ps, st, rng, x_shape, batch_size),
        :ddpm => (gen_compiled, ps, st, rng, x_shape, batch_size) -> generate_batch_ddpm(model, gen_compiled, ps, st, rng, x_shape, batch_size),
        :pang => (gen_compiled, ps, st, rng, x_shape, batch_size) -> generate_batch_pang(model, gen_compiled, ps, st, rng, x_shape, batch_size),
    )

    prepare_batch_fn = prepare_batch_fns[model_type]
    call_train_step_fn = call_train_step_fns[model_type]
    generate_batch_fn = generate_batch_fns[model_type]

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
        rng,
        prepare_batch_fn,
        call_train_step_fn,
        generate_batch_fn,
    )

end

image_test_loss(x, x_recon) = Flux.mse(x, x_recon)

function train!(t::Trainer)
    ps = t.ps
    st = t.st
    opt_state = t.opt_state
    opt_state_gen = t.opt_state_gen
    opt_state_disc = t.opt_state_disc

    function save_checkpoint(epoch::Int)
        return jldsave(
            t.file_loc * "ckpt_epoch_$(epoch).jld2";
            params = Array(ps),
            state = st |> cpu_device(),
        )
    end

    function log_loss(
            now_time::Float64,
            epoch::Int,
            train_loss::Float32,
            test_loss::Float32 = 0.0f0
        )
        loss_file = t.file_loc * "loss.csv"
        return open(loss_file, "a") do file
            write(file, "$now_time,$epoch,$train_loss,$test_loss\n")
        end
    end

    function prepare_batch(x, train_idx = nothing)
        return t.prepare_batch_fn(
            pu(x),
            train_idx,
            t.rng,
            t.x_shape
        )
    end

    function call_train_step(batch_args)
        return t.call_train_step_fn(
            t.train_step,
            opt_state,
            opt_state_gen,
            opt_state_disc,
            ps,
            st,
            batch_args
        )
    end

    function generate_batch()
        x_gen, st_new = t.generate_batch_fn(
            t.gen_compiled,
            ps,
            st,
            t.rng,
            t.x_shape,
            t.batch_size
        )
        return x_gen, st_new
    end


    function save_generated_images(gen_data, epoch; final::Bool = false)
        filename = final ? "generated_images.h5" : "generated_images_epoch_$(epoch).h5"
        return try
            h5write(t.file_loc * filename, "samples", gen_data)
        catch
            rm(t.file_loc * filename)
            h5write(t.file_loc * filename, "samples", gen_data)
        end
    end

    train_idx_start = 1
    x_sample = first(t.train_loader) |> pu
    x_gen_sample, st = generate_batch()
    test_step_compiled = Reactant.@compile image_test_loss(x_sample, x_gen_sample)

    function compute_test()
        test_loss = 0.0f0
        for x in t.test_loader
            x_gen, st = generate_batch()
            test_loss += test_step_compiled(pu(x), x_gen) |> Float32
        end
        return test_loss / length(t.test_loader)
    end

    num_batches = length(t.train_loader)
    start_time = time()
    train_idx = train_idx_start

    for epoch in 1:t.N_epochs
        train_loss = 0.0f0

        for (batch_idx, x) in enumerate(t.train_loader)
            batch_args = prepare_batch(x, train_idx)
            loss, ps, opt_state, opt_state_gen, opt_state_disc, st = call_train_step(batch_args)
            train_loss += Float32(loss)
            train_idx += 1
        end

        train_loss /= num_batches

        test_loss = compute_test()
        now_time = time() - start_time

        println(
            "Epoch: $epoch, Train Loss: $train_loss, Test Loss: $test_loss"
        )
        log_loss(now_time, epoch, train_loss, test_loss)

        if t.gen_every > 0 && epoch % t.gen_every == 0
            num_batches_gen = fld(t.num_generated_samples, 10) ÷ t.batch_size # Save 1/10 of the samples to conserve space

            if num_batches_gen > 0
                first_batch, st = generate_batch()
                first_batch = Array(first_batch)
                concat_dim = length(size(first_batch))
                batches_to_cat = Vector{typeof(first_batch)}()
                sizehint!(batches_to_cat, num_batches_gen)
                push!(batches_to_cat, first_batch)

                for _ in 2:num_batches_gen
                    batch, st = generate_batch()
                    push!(batches_to_cat, Array(batch))
                end

                gen_data = cat(batches_to_cat..., dims = concat_dim)
                save_generated_images(gen_data, epoch)
            end
        end

        if t.checkpoint_every > 0 && epoch % t.checkpoint_every == 0
            save_checkpoint(epoch)
        end

        GC.gc()
    end

    t.ps = ps
    t.st = st
    t.opt_state = opt_state
    t.opt_state_gen = opt_state_gen
    t.opt_state_disc = opt_state_disc

    num_batches_gen = t.num_generated_samples ÷ t.batch_size
    return if num_batches_gen > 0
        first_batch, st = generate_batch()
        first_batch = Array(first_batch)
        concat_dim = length(size(first_batch))
        batches_to_cat = Vector{typeof(first_batch)}()
        sizehint!(batches_to_cat, num_batches_gen)
        push!(batches_to_cat, first_batch)

        for _ in 2:num_batches_gen
            batch, st = generate_batch()
            push!(batches_to_cat, Array(batch))
            GC.gc()
        end

        gen_data = cat(batches_to_cat..., dims = concat_dim)
        save_generated_images(gen_data, t.N_epochs; final = true)
    end
end

end
