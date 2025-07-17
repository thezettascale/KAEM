module trainer

export T_KAM_trainer, init_trainer, train!

include("../T-KAM/T-KAM.jl")
include("../T-KAM/kan/grid_updating.jl")
include("optimizer.jl")
include("../utils.jl")
include("data_utils.jl")
using .T_KAM_model
using .GridUpdating: update_model_grid
using .optimization
using .Utils: device, half_quant, full_quant, hq, fq
using .DataUtils: get_vision_dataset, get_text_dataset
using Flux: onecold, mse

using CUDA, KernelAbstractions, Tullio
using Random, ComponentArrays, CSV, HDF5, JLD2, ConfParser
using Optimization, OptimizationOptimJL, Lux, LuxCUDA, LinearAlgebra, Accessors
using Enzyme

mutable struct T_KAM_trainer{T<:half_quant,U<:full_quant}
    model::Any
    cnn::Bool
    o::opt
    dataset_name::AbstractString
    ps::ComponentArray
    st::NamedTuple
    N_epochs::Int
    train_loader_state::Tuple{Any,Int}
    x::AbstractArray{T}
    num_generated_samples::Int
    batch_size_for_gen::Int
    grid_update_frequency::Int
    last_grid_update::Int
    save_model::Bool
    gen_type::AbstractString
    checkpoint_every::Int
    loss::full_quant
    rng::AbstractRNG
end

function init_trainer(
    rng::AbstractRNG,
    conf::ConfParse,
    dataset_name;
    img_resize = nothing,
    file_loc = nothing,
    save_model = true,
)

    # Load dataset
    N_train = parse(Int, retrieve(conf, "TRAINING", "N_train"))
    N_test = parse(Int, retrieve(conf, "TRAINING", "N_test"))
    num_generated_samples = parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"))
    batch_size_for_gen = parse(Int, retrieve(conf, "TRAINING", "batch_size_for_gen"))
    # cnn = dataset_name == "CIFAR10" || dataset_name == "SVHN" 
    seq = dataset_name == "PTB" || dataset_name == "SMS_SPAM"
    # cnn = false
    gen_type = seq ? "logits" : "images"
    # commit!(conf, "CNN", "use_cnn_lkhood", string(cnn))
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
        parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_langevin")) ? "autoMALA" :
        "importance"

    if mala == "autoMALA" && !parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_autoMALA"))
        mala = "ULA_mixture"
    end


    model_type = N_t > 1 ? "thermo/$(dataset_name)" : "vanilla/$(dataset_name)"

    prior_spline_fcn =
        "importance" *
        "_" *
        retrieve(conf, "EbmModel", "π_0") *
        "_" *
        retrieve(conf, "EbmModel", "spline_function")
    if dataset_name in ["DARCY_PERM", "DARCY_FLOW", "MNIST", "FMNIST"]
        model_type = model_type * "/" * prior_spline_fcn
    end

    if length(parse.(Int, retrieve(conf, "EbmModel", "layer_widths"))) > 2
        model_type = model_type * "/deep"
    elseif parse(Bool, retrieve(conf, "EbmModel", "mixture_model"))
        model_type = model_type * "/mixture"
    else
        model_type = model_type * "/univariate"
    end

    file_loc = isnothing(file_loc) ? "logs/$(model_type)/" : file_loc
    mkpath(file_loc)

    # Initialize model
    println("Initializing model")
    model = init_T_KAM(dataset, conf, x_shape; file_loc = file_loc, rng = rng)
    params, state = Lux.setup(rng, model)
    model = move_to_hq(model)
    model = prep_model(model, params, state, x; rng = rng)

    optimizer = create_opt(conf)
    grid_update_frequency =
        parse(Int, retrieve(conf, "GRID_UPDATING", "grid_update_frequency"))

    N_epochs = parse(Int, retrieve(conf, "TRAINING", "N_epochs"))
    x, loader_state = iterate(model.train_loader)
    checkpoint_every = parse(Int, retrieve(conf, "TRAINING", "checkpoint_every"))

    try
        h5write(file_loc * "real_$(gen_type).h5", "samples", Float32.(save_dataset))
    catch
        rm(file_loc * "real_$(gen_type).h5")
        h5write(file_loc * "real_$(gen_type).h5", "samples", Float32.(save_dataset))
    end

    open(file_loc * "loss.csv", "w") do file
        write(file, "Time (s),Epoch,Train MLE Loss,Test MSE Loss,Grid Updated\n")
    end

    return T_KAM_trainer(
        model,
        cnn,
        optimizer,
        dataset_name,
        device(params),
        device(state),
        N_epochs,
        loader_state,
        device(x),
        num_generated_samples,
        batch_size_for_gen,
        grid_update_frequency,
        1,
        save_model,
        gen_type,
        checkpoint_every,
        zero(full_quant),
        rng,
    )
end

function train!(t::T_KAM_trainer)

    # (Move off GPU)
    @reset t.st.train_idx = t.st.train_idx |> cpu_device()
    @reset t.st.η_init = t.st.η_init |> cpu_device()
    loss_scaling = t.model.loss_scaling |> full_quant

    num_batches = length(t.model.train_loader)
    grid_updated = 0
    num_param_updates = num_batches * t.N_epochs
    grads = half_quant.(Enzyme.make_zero(half_quant.(t.ps)))

    loss_file = t.model.file_loc * "loss.csv"

    function find_nan(grads)
        for k in keys(grads)
            if any(isnan, grads[k]) || any(isinf, grads[k])
                for i in keys(grads[k])
                    any(isnan.(grads[k][i])) ||
                        any(isinf.(grads[k][i])) && error("NaN/Inf in $k, $i gradients")
                end
            end
        end
    end

    # Gradient for a single batch
    function grad_fcn(G, u, args...)
        t.ps = u

        # Grid updating for likelihood model
        if (
            t.st.train_idx == 1 ||
            (t.st.train_idx - t.last_grid_update >= t.grid_update_frequency)
        ) && (t.model.update_llhood_grid || t.model.update_prior_grid)
            t.model, t.ps, t.st =
                update_model_grid(t.model, t.x, t.ps, Lux.testmode(t.st); rng = t.rng)
            t.grid_update_frequency =
                t.st.train_idx > 1 ?
                floor(
                    t.grid_update_frequency *
                    (2 - t.model.grid_update_decay)^t.st.train_idx,
                ) : t.grid_update_frequency
            t.last_grid_update = t.st.train_idx
            grid_updated = 1

            t.model.verbose && println("Iter: $(t.st.train_idx), Grid updated")
        end

        # Reduced precision grads, (switches to full precision for accumulation, not forward passes)
        loss, grads, st_ebm, st_gen = t.model.loss_fcn(
            half_quant(t.ps),
            grads,
            Lux.trainmode(t.st),
            t.model,
            t.x;
            rng = t.rng,
        )
        t.loss = full_quant(loss) / loss_scaling
        copy!(G, full_quant.(grads) ./ loss_scaling)
        @reset t.st.ebm = st_ebm
        @reset t.st.gen = st_gen

        isnan(norm(G)) || isinf(norm(G)) && find_nan(G)
        t.model.verbose && println("Iter: $(t.st.train_idx), Grad norm: $(norm(G))")
        return G
    end

    train_loss = 0

    # Train and test loss with logging
    function opt_loss(u, args...)
        t.ps = u
        train_loss += t.loss

        # After one epoch, calculate test loss and log to CSV
        if t.st.train_idx % num_batches == 0 || t.st.train_idx == 1

            test_loss = 0
            for x in t.model.test_loader
                x_gen, st_ebm, st_gen = CUDA.@fastmath generate_batch(
                    t.model,
                    t.ps,
                    Lux.testmode(t.st),
                    size(x)[end];
                    rng = t.rng,
                )
                @reset t.st.ebm = st_ebm
                @reset t.st.gen = st_gen
                x_gen = x_gen .|> full_quant

                # MSE loss between pixels for images, and max index for logits
                if t.gen_type == "logits"
                    idxs = dropdims(argmax(x_gen, dims = 1); dims = 1)
                    test_loss +=
                        sum((device(onecold(x, 1:size(x, 1))) .- getindex.(idxs, 1)) .^ 2) /
                        size(x)[end]
                else
                    test_loss += mse(device(x), x_gen)
                end
            end

            train_loss = train_loss / num_batches
            test_loss /= length(t.model.test_loader)
            now_time = time() - start_time
            epoch = t.st.train_idx == 1 ? 0 : fld(t.st.train_idx, num_batches)

            m.verbose && println(
                "Epoch: $(epoch), Train Loss: $(train_loss), Test Loss: $(test_loss)",
            )

            open(loss_file, "a") do file
                write(file, "$now_time,$(epoch),$train_loss,$test_loss,$grid_updated\n")
            end

            (t.checkpoint_every > 0 && epoch % t.checkpoint_every) == 0 && jldsave(
                t.model.file_loc * "ckpt_epoch_$(epoch).jld2";
                params = t.ps |> cpu_device(),
                state = t.st |> cpu_device(),
                rng = t.rng,
            )

            train_loss = 0
            grid_updated = 0

            # Save images
            gen_data = zeros(half_quant, t.model.lkhood.x_shape..., 0)
            idx = length(t.model.lkhood.x_shape) + 1
            for i = 1:(t.num_generated_samples//t.batch_size_for_gen)
                batch, st_ebm, st_gen = CUDA.@fastmath generate_batch(
                    t.model,
                    t.ps,
                    Lux.testmode(t.st),
                    t.batch_size_for_gen;
                    rng = t.rng,
                )
                @reset t.st.ebm = st_ebm
                @reset t.st.gen = st_gen
                gen_data = cat(gen_data, cpu_device()(batch), dims = idx)
            end

            try
                h5write(
                    t.model.file_loc * "generated_$(t.gen_type)_epoch_$(epoch).h5",
                    "samples",
                    Float32.(gen_data),
                )
            catch
                rm(t.model.file_loc * "generated_$(t.gen_type)_epoch_$(epoch).h5")
                h5write(
                    t.model.file_loc * "generated_$(t.gen_type)_epoch_$(epoch).h5",
                    "samples",
                    Float32.(gen_data),
                )
            end

        end

        @reset t.st.train_idx += 1

        # Iterate loader, reset to first batch when epoch ends
        x, t.train_loader_state =
            (t.st.train_idx % num_batches == 0) ? iterate(t.model.train_loader) :
            iterate(t.model.train_loader, t.train_loader_state)
        t.x = device(x)
        return loss
    end

    start_time = time()

    optf = Optimization.OptimizationFunction(opt_loss; grad = grad_fcn)
    optprob = Optimization.OptimizationProblem(optf, copy(t.ps))

    # Optimization only stops when maxiters is reached
    res = Optimization.solve(
        optprob,
        t.o.init_optimizer();
        maxiters = num_param_updates,
        verbose = true,
        abstol = -one(full_quant),
        reltol = -one(full_quant),
        x_tol = -one(full_quant),
        x_abstol = -one(full_quant),
        x_reltol = -one(full_quant),
        f_tol = -one(full_quant),
        f_abstol = -one(full_quant),
        f_reltol = -one(full_quant),
        g_tol = -one(full_quant),
        g_abstol = -one(full_quant),
        g_reltol = -one(full_quant),
        outer_x_abstol = -one(full_quant),
        outer_x_reltol = -one(full_quant),
        outer_f_abstol = -one(full_quant),
        outer_f_reltol = -one(full_quant),
        outer_g_abstol = -one(full_quant),
        outer_g_reltol = -one(full_quant),
        successive_f_tol = num_param_updates,
        allow_f_increases = true,
        allow_outer_f_increases = true,
    )

    t.ps = res.minimizer

    # Generate samples
    gen_data = zeros(half_quant, t.model.lkhood.x_shape..., 0)
    idx = length(t.model.lkhood.x_shape) + 1
    for i = 1:(t.num_generated_samples//t.batch_size_for_gen)
        batch, t.st = CUDA.@fastmath generate_batch(
            t.model,
            t.ps,
            Lux.testmode(t.st),
            t.batch_size_for_gen;
            rng = t.rng,
        )
        gen_data = cat(gen_data, cpu_device()(batch), dims = idx)
    end

    try
        h5write(
            t.model.file_loc * "generated_$(t.gen_type).h5",
            "samples",
            Float32.(gen_data),
        )
    catch
        rm(t.model.file_loc * "generated_$(t.gen_type).h5")
        h5write(
            t.model.file_loc * "generated_$(t.gen_type).h5",
            "samples",
            Float32.(gen_data),
        )
    end

    t.save_model && jldsave(
        t.model.file_loc * "saved_model.jld2";
        params = t.ps |> cpu_device(),
        state = t.st |> cpu_device(),
    )
end

end
