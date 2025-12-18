module trainer

export KAEM_trainer, init_trainer, train!

using Flux: onecold, mse
using ImageQualityIndexes: assess_msssim
using Random, ComponentArrays, CSV, HDF5, JLD2, ConfParser, Reactant
using Lux, LinearAlgebra, Accessors
using MultivariateStats: reconstruct
using MLDataDevices: cpu_device
using ParameterSchedulers: next!
using Optimisers: adjust!
using HyperTuning: report_value!, should_prune

include("../utils.jl")
using .Utils

include("../KAEM/KAEM.jl")
using .KAEM_model

include("../KAEM/model_setup.jl")
using .ModelSetup

include("../KAEM/grid_updating.jl")
using .ModelGridUpdating

include("optimizer.jl")
include("data_utils.jl")
using .optimization
using .DataUtils: get_vision_dataset, get_text_dataset

include("../KAEM/rng.jl")
using .HLOrng

mutable struct KAEM_trainer{T <: Float32}
    model::Any
    grid_updater::Any
    cnn::Bool
    opt_state::Any
    schedule::Any
    dataset_name::AbstractString
    ps::ComponentArray{T}
    st_kan::NamedTuple
    st_lux::NamedTuple
    st_rng::NamedTuple
    N_epochs::Int
    train_loader_state::Tuple{Any, Int}
    x::AbstractArray{T}
    num_generated_samples::Int
    last_grid_update::Int
    save_model::Bool
    img_tuning::Bool
    gen_type::AbstractString
    checkpoint_every::Int
    gen_every::Int
    loss::T
    rng::AbstractRNG
end

function init_trainer(
        rng::AbstractRNG,
        conf::ConfParse,
        dataset_name;
        img_resize = nothing,
        file_loc = nothing,
        save_model = true,
        img_tuning = false,
    )

    N_train = parse(Int, retrieve(conf, "TRAINING", "N_train"))
    N_test = parse(Int, retrieve(conf, "TRAINING", "N_test"))
    num_generated_samples = parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"))
    seq = dataset_name == "PTB" || dataset_name == "SMS_SPAM"
    gen_type = seq ? "logits" : "images"
    cnn = parse(Bool, retrieve(conf, "CNN", "use_cnn_lkhood"))
    sequence_length = seq ? parse(Int, retrieve(conf, "SEQ", "sequence_length")) : 0
    commit!(conf, "SEQ", "sequence_length", string(sequence_length)) # Make sure 0 is set if not sequence
    vocab_size = parse(Int, retrieve(conf, "SEQ", "vocab_size"))
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    dataset, x_shape, save_dataset = (
        seq ?
            get_text_dataset(
                dataset_name,
                N_train,
                N_test,
                num_generated_samples;
                sequence_length = sequence_length,
                vocab_size = vocab_size,
                batch_size = batch_size,
            ) :
            get_vision_dataset(
                dataset_name,
                N_train,
                N_test,
                num_generated_samples;
                img_resize = img_resize,
                cnn = cnn,
            )
    )

    println("Dataset loaded")

    # Log against ULA and autoMALA
    N_t = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps"))
    mala =
        parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_langevin")) ? "ULA" :
        "importance"

    model_type =
        N_t > 1 ? "Thermodynamic/$(dataset_name)/$(mala)" :
        "Vanilla/$(dataset_name)/$(mala)"

    prior_spline_fcn =
        retrieve(conf, "EbmModel", "π_0") *
        "_" *
        retrieve(conf, "EbmModel", "spline_function")
    if mala == "importance"
        model_type = model_type * "/" * prior_spline_fcn
    end

    if length(parse.(Int, retrieve(conf, "EbmModel", "layer_widths"))) > 2
        model_type = model_type * "/deep"
    elseif parse(Bool, retrieve(conf, "MixtureModel", "use_mixture_prior"))
        model_type = model_type * "/mixture"
    else
        model_type = model_type * "/univariate"
    end

    file_loc = isnothing(file_loc) ? "logs/$(model_type)/" : file_loc
    !img_tuning && mkpath(file_loc)

    println("Initializing model...")
    model = init_KAEM(dataset, conf, x_shape; file_loc = file_loc, rng = rng)
    x, loader_state = iterate(model.train_loader)
    x = pu(x)
    optimizer = create_opt(conf)

    model, opt_state, params, st_kan, st_lux, st_rng = prep_model(model, x, optimizer; rng = rng)


    N_epochs = parse(Int, retrieve(conf, "TRAINING", "N_epochs"))
    checkpoint_every = parse(Int, retrieve(conf, "TRAINING", "checkpoint_every"))
    gen_every = parse(Int, retrieve(conf, "TRAINING", "gen_every"))

    if !img_tuning
        try
            h5write(file_loc * "real_$(gen_type).h5", "samples", save_dataset)
        catch
            rm(file_loc * "real_$(gen_type).h5")
            h5write(file_loc * "real_$(gen_type).h5", "samples", save_dataset)
        end

        open(file_loc * "loss.csv", "w") do file
            write(file, "Time (s),Epoch,Train MLE Loss,Test MSE Loss,Grid Updated\n")
        end
    end

    return KAEM_trainer(
        model,
        GridUpdater(model, conf),
        cnn,
        opt_state,
        optimizer.schedule,
        dataset_name,
        params,
        st_kan,
        st_lux,
        st_rng,
        N_epochs,
        loader_state,
        x,
        num_generated_samples,
        1,
        save_model,
        img_tuning,
        gen_type,
        checkpoint_every,
        gen_every,
        zero(Float32),
        rng,
    )
end

function logit_test_loss(x, x_gen)
    idxs = dropdims(argmax(x_gen, dims = 1); dims = 1)
    return sum(
        (
            onecold(x, 1:size(x, 1)) .- getindex.(idxs, 1)
        ) .^ 2
    ) / size(x)[end]
end

function image_test_loss(x, x_gen)
    return mse(x, x_gen)
end

function train!(t::KAEM_trainer; train_idx::Int = 1, trial = nothing)
    num_batches = length(t.model.train_loader)
    grid_updated = 0
    num_param_updates = num_batches * t.N_epochs
    loss_file = t.model.file_loc * "loss.csv"

    (isnothing(trial) && t.img_tuning) && error("Must provide trial when tuning")

    grid_compiled = Reactant.@compile t.grid_updater(
        t.x,
        t.ps,
        t.st_kan,
        Lux.testmode(t.st_lux),
        train_idx,
        t.st_rng,
    )

    gen_compiled = Reactant.@compile t.model(
        t.ps,
        t.st_kan,
        Lux.testmode(t.st_lux),
        t.st_rng,
    )

    test_step = t.gen_type == "logits" ? logit_test_loss : image_test_loss
    x_gen = first(
        gen_compiled(
            t.ps,
            t.st_kan,
            Lux.testmode(t.st_lux),
            t.st_rng
        )
    )

    test_step = Reactant.@compile test_step(t.x, x_gen)

    real_ssim_x = zeros(Float32, t.model.lkhood.x_shape..., 0)
    if t.img_tuning
        for x in t.model.test_loader
            real_ssim_x = cat(real_ssim_x, x; dims = length(t.model.lkhood.x_shape) + 1)
        end
    end


    # Update for a single batch
    function step!()
        t.st_rng = seed_rand(t.model; rng = t.rng)

        if (
                train_idx == 1 || (train_idx - t.last_grid_update >= t.grid_updater.update_frequency)
            ) && (t.grid_updater.update_llhood_grid || t.grid_updater.update_prior_grid)
            t.ps, t.st_kan, t.st_lux = grid_compiled(
                t.x,
                t.ps,
                t.st_kan,
                Lux.testmode(t.st_lux),
                train_idx,
                t.st_rng,
            )

            grid_updated = 1
            t.last_grid_update = train_idx
            if train_idx > 1
                @reset t.grid_updater.update_frequency = (
                    floor(
                        t.grid_updater.update_frequency *
                            (2 - t.grid_updater.update_decay)^train_idx
                    )
                )
            end

            t.model.verbose && println("Iter: $(train_idx), Grid updated")
        end

        t.loss, t.ps, t.opt_state, st_ebm, st_gen = t.model.train_step(
            t.opt_state,
            t.ps,
            t.st_kan,
            t.st_lux,
            t.x,
            train_idx,
            t.st_rng,
        )
        @reset t.st_lux.ebm = st_ebm
        @reset t.st_lux.gen = st_gen

        if isnan(Float32(t.loss))
            train_idx = Inf
        end

        if t.model.verbose && !t.img_tuning
            println("Iter: $(train_idx), Loss: $(t.loss)")
        end

        return nothing
    end

    train_loss = 0

    # Train and test loss with logging
    function opt_loss!()
        train_loss += t.loss
        epoch = train_idx == 1 ? 0 : fld(train_idx, num_batches)

        # After one epoch, calculate test loss and log to CSV
        first_bool = !t.img_tuning && train_idx == 1
        epoch_done = train_idx % num_batches == 0
        if !t.img_tuning && epoch_done || first_bool
            test_loss = 0.0e0
            for x in t.model.test_loader
                t.st_rng = seed_rand(t.model; rng = t.rng)
                x_gen, st_ebm, st_gen = gen_compiled(
                    t.ps,
                    t.st_kan,
                    Lux.testmode(t.st_lux),
                    t.st_rng,
                )
                @reset t.st_lux.ebm = st_ebm
                @reset t.st_lux.gen = st_gen

                test_loss += test_step(pu(x), x_gen) |> Float64
            end

            train_loss = train_loss / num_batches
            test_loss /= length(t.model.test_loader)

            now_time = time() - start_time

            t.model.verbose && println(
                "Epoch: $(epoch), Train Loss: $(train_loss), Test Loss: $(test_loss)",
            )

            open(loss_file, "a") do file
                write(file, "$now_time,$(epoch),$train_loss,$test_loss,$grid_updated\n")
            end

            train_loss = 0.0f0
            grid_updated = 0
        end

        if (t.gen_every > 0) && (epoch % t.gen_every == 0) && epoch_done && t.img_tuning
            gen_ssim_x = zeros(Float32, t.model.lkhood.x_shape..., 0)
            for x in t.model.test_loader
                t.st_rng = seed_rand(t.model; rng = t.rng)
                x_gen, st_ebm, st_gen = gen_compiled(
                    t.ps,
                    t.st_kan,
                    Lux.testmode(t.st_lux),
                    t.st_rng,
                )
                @reset t.st_lux.ebm = st_ebm
                @reset t.st_lux.gen = st_gen

                gen_ssim_x = cat(
                    gen_ssim_x,
                    Array(x_gen);
                    dims = length(t.model.lkhood.x_shape) + 1
                )
            end

            ssim = (1.0f0 - assess_msssim(real_ssim_x, gen_ssim_x)) |> Float64
            ssim /= length(t.model.test_loader)

            println("Epoch: ", epoch, " MS-SSIM: ", ssim)
            report_value!(trial, Float64(mse(real_ssim_x, gen_ssim_x)) + ssim)
            if should_prune(trial)
                train_idx = Inf
            end
        end

        if (t.checkpoint_every > 0) && (epoch % t.checkpoint_every == 0) && epoch_done && !t.img_tuning
            jldsave(
                t.model.file_loc * "ckpt_epoch_$(epoch).jld2";
                params = Array(t.ps),
                kan_state = t.st_kan |> cpu_device(),
                lux_state = t.st_lux |> cpu_device(),
                rng = t.rng,
            )
        end

        # Save images - collect batches first then concatenate once to avoid O(n²) allocations
        if (t.gen_every > 0) && (epoch % t.gen_every == 0) && epoch_done && !t.img_tuning
            num_batches_to_save = fld(t.num_generated_samples, 10) ÷ t.model.batch_size # Save 1/10 of the samples to conserve space
            if num_batches_to_save > 0
                concat_dim = length(t.model.lkhood.x_shape) + 1
                t.st_rng = seed_rand(t.model; rng = t.rng)

                # Get first batch to determine type
                first_batch, st_ebm, st_gen = gen_compiled(
                    t.ps,
                    t.st_kan,
                    Lux.testmode(t.st_lux),
                    t.st_rng,
                )
                @reset t.st_lux.ebm = st_ebm
                @reset t.st_lux.gen = st_gen
                first_batch = Array(first_batch)

                batches_to_cat = Vector{typeof(first_batch)}()
                sizehint!(batches_to_cat, num_batches_to_save)
                push!(batches_to_cat, first_batch)

                for i in 2:num_batches_to_save
                    t.st_rng = seed_rand(t.model; rng = t.rng)
                    batch, st_ebm, st_gen = gen_compiled(
                        t.ps,
                        t.st_kan,
                        Lux.testmode(t.st_lux),
                        t.st_rng,
                    )
                    @reset t.st_lux.ebm = st_ebm
                    @reset t.st_lux.gen = st_gen
                    push!(batches_to_cat, Array(batch))
                end
                gen_data = cat(batches_to_cat..., dims = concat_dim)
            else
                gen_data = zeros(Float32, t.model.lkhood.x_shape..., 0)
            end

            if !t.model.lkhood.SEQ && !t.model.lkhood.CNN && t.model.use_pca
                gen_data = reconstruct(t.model.PCA_model, gen_data)
                gen_data = (
                    reshape(
                        gen_data,
                        t.model.original_data_size...,
                        size(gen_data)[end],
                    ),
                )
            end

            try
                h5write(
                    t.model.file_loc * "generated_$(t.gen_type)_epoch_$(epoch).h5",
                    "samples",
                    gen_data,
                )
            catch
                rm(t.model.file_loc * "generated_$(t.gen_type)_epoch_$(epoch).h5")
                h5write(
                    t.model.file_loc * "generated_$(t.gen_type)_epoch_$(epoch).h5",
                    "samples",
                    gen_data,
                )
            end
        end

        train_idx += 1

        # Iterate loader, reset to first batch when epoch ends
        x, t.train_loader_state =
            (train_idx % num_batches == 0) ? iterate(t.model.train_loader) :
            iterate(t.model.train_loader, t.train_loader_state)
        t.x = pu(x)

        return nothing
    end

    start_time = time()

    while train_idx <= num_param_updates
        η = next!(t.schedule)
        adjust!(t.opt_state, η)
        step!()
        opt_loss!()
        GC.gc()
    end

    # Generate samples
    num_batches = t.num_generated_samples ÷ t.model.batch_size
    concat_dim = length(t.model.lkhood.x_shape) + 1
    first_batch, st_ebm, st_gen = gen_compiled(
        t.ps,
        t.st_kan,
        Lux.testmode(t.st_lux),
        t.st_rng,
    )
    first_batch = Array(first_batch)
    batches_to_cat = Vector{typeof(first_batch)}()
    sizehint!(batches_to_cat, num_batches)
    push!(batches_to_cat, first_batch)

    for i in 2:num_batches
        t.st_rng = seed_rand(t.model; rng = t.rng)
        batch, st_ebm, st_gen = gen_compiled(
            t.ps,
            t.st_kan,
            Lux.testmode(t.st_lux),
            t.st_rng,
        )
        push!(batches_to_cat, Array(batch))
    end

    gen_data = cat(batches_to_cat..., dims = concat_dim)

    if !t.model.lkhood.SEQ && !t.model.lkhood.CNN && t.model.use_pca
        gen_data = reconstruct(t.model.PCA_model, gen_data)
        gen_data =
            (reshape(gen_data, t.model.original_data_size..., size(gen_data)[end]))
    end

    if !t.img_tuning
        try
            h5write(t.model.file_loc * "generated_$(t.gen_type).h5", "samples", gen_data)
        catch
            rm(t.model.file_loc * "generated_$(t.gen_type).h5")
            h5write(t.model.file_loc * "generated_$(t.gen_type).h5", "samples", gen_data)
        end
    end

    (t.save_model && !t.img_tuning) && jldsave(
        t.model.file_loc * "saved_model.jld2";
        params = Array(t.ps),
        kan_state = t.st_kan |> cpu_device(),
        lux_state = t.st_lux |> cpu_device(),
        train_idx = train_idx,
    )

    ssim = 0.0e0
    if t.img_tuning
        gen_ssim_x = zeros(Float32, t.model.lkhood.x_shape..., 0)
        for x in t.model.test_loader
            t.st_rng = seed_rand(t.model; rng = t.rng)
            x_gen, st_ebm, st_gen = gen_compiled(
                t.ps,
                t.st_kan,
                Lux.testmode(t.st_lux),
                t.st_rng,
            )
            @reset t.st_lux.ebm = st_ebm
            @reset t.st_lux.gen = st_gen
            gen_ssim_x = cat(
                gen_ssim_x,
                Array(x_gen);
                dims = length(t.model.lkhood.x_shape) + 1
            )
        end
        ssim = (1.0f0 - assess_msssim(real_ssim_x, gen_ssim_x)) |> Float64
        ssim /= length(t.model.test_loader)
        ssim += Float64(mse(real_ssim_x, gen_ssim_x))
    end

    return ssim
end

end
