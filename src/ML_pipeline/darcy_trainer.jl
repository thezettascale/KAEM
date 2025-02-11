module trainer

export T_KAM_trainer, init_trainer, train!

include("../T-KAM/T-KAM.jl")
include("optimizer.jl")
include("../utils.jl")
include("data_utils.jl")
using .T_KAM_model
using .optimization
using .Utils: device, half_quant, full_quant, hq, fq

using CUDA, KernelAbstractions, Tullio
using Random, ComponentArrays, CSV, HDF5, JLD2, ConfParser
using Optimization, OptimizationOptimJL, Lux, LuxCUDA, LinearAlgebra, Accessors
using Zygote: withgradient
using Flux: DataLoader

mutable struct T_KAM_trainer
    model
    o::opt
    ps::ComponentArray
    st::NamedTuple
    N_epochs::Int
    train_loader_state::Tuple{Any, Int}
    x::AbstractArray{full_quant}
    num_generated_samples::Int
    batch_size_for_gen::Int
    seed::Int
    grid_update_frequency::Int
    last_grid_update::Int
    save_model::Bool
    loss::full_quant
    checkpoint::Bool
    test_loader::DataLoader
end

function init_trainer(rng::AbstractRNG, conf::ConfParse, informed::Bool; 
    seed=1, file_loc=nothing, save_model=true)

    # Load dataset
    N_train = parse(Int, retrieve(conf, "TRAINING", "N_train"))
    N_test = parse(Int, retrieve(conf, "TRAINING", "N_test"))

    N_train > 1000 || N_test > 1000 && error("There are only 1000 examples in Darcy tain/test. Please reduce your config.")

    num_generated_samples = parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"))
    batch_size_for_gen = parse(Int, retrieve(conf, "TRAINING", "batch_size_for_gen"))
    gen_type = "pressure_fields"
    commit!(conf, "CNN", "use_cnn_lkhood", "false")
    commit!(conf, "SEQ", "sequence_length", "-1") 
    commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

    if informed
        commit!(conf, "MIX_PRIOR", "π_0", "lognormal")
        commit!(conf, "MIX_PRIOR", "spline_function", "FFT")
        commit!(conf, "KAN_LIKELIHOOD", "spline_function", "FFT")
    else
        commit!(conf, "MIX_PRIOR", "π_0", "uniform")
        commit!(conf, "MIX_PRIOR", "spline_function", "RBF")
        commit!(conf, "KAN_LIKELIHOOD", "spline_function", "RBF")
    end

    model_type = informed ? "informed" : "uninformed"
    file_loc = isnothing(file_loc) ? "logs/Darcy/$(model_type)/seed_$(seed)/" : file_loc
    mkpath(file_loc)

    f_train = h5open("PDE_data/darcy_32/darcy_train_32.h5")
    f_test = h5open("PDE_data/darcy_32/darcy_test_32.h5")

    dataset = cat(f_train["y"][:,:,1:N_train], f_test["y"][:,:,1:N_test], dims=3)
    dataset = reshape(dataset, 32, 32, 1, :)
    x_shape = (32, 32, 1)

    aux_dataset_train = reshape(f_train["x"][:,:,1:N_train], 32*32, :) .|> half_quant
    test_x = reshape(f_test["x"][:,:,1:N_test], 32, 32, 1, N_test) 

    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
    test_loader = DataLoader((test_x, dataset[:,:,:,N_train+1:end]), batchsize=batch_size, shuffle=true)

    # Initialize model
    model = init_T_KAM(dataset, conf, x_shape; file_loc=file_loc, prior_seed=seed, lkhood_seed=seed, data_seed=seed, aux_data=aux_dataset_train)
    params, state = Lux.setup(rng, model)
    model = move_to_hq(model)

    optimizer = create_opt(conf)
    grid_update_frequency = parse(Int, retrieve(conf, "GRID_UPDATING", "grid_update_frequency"))

    N_epochs = parse(Int, retrieve(conf, "TRAINING", "N_epochs"))
    x, loader_state = iterate(model.train_loader) 
    checkpoint = parse(Bool, retrieve(conf, "TRAINING", "checkpoint"))

    open(file_loc * "loss.csv", "w") do file
        write(file, "Time (s),Epoch,Train MLE Loss,Test Gen Loss,Test Recon Loss,Grid Updated\n")
    end
    
    return T_KAM_trainer(
        model, 
        optimizer, 
        device(params), 
        device(state), 
        N_epochs, 
        loader_state, 
        device(x), 
        num_generated_samples,
        batch_size_for_gen,
        seed, 
        grid_update_frequency,
        1,
        save_model,
        full_quant(0),
        checkpoint,
        test_loader
    )
end

function train!(t::T_KAM_trainer)
    
    # (Move off GPU)
    @reset t.st.train_idx = t.st.train_idx |> cpu_device()
    @reset t.st.η_init = t.st.η_init |> cpu_device()

    num_batches = length(t.model.train_loader)
    grid_updated = 0
    num_param_updates = num_batches * t.N_epochs
    
    loss_file = t.model.file_loc * "loss.csv"

    function find_nan(grads)
        for k in keys(grads)
            if any(isnan, grads[k]) || any(isinf, grads[k])
                for i in keys(grads[k])
                    any(isnan.(grads[k][i])) || any(isinf.(grads[k][i])) && error("NaN/Inf in $k, $i gradients")
                end
            end
        end
    end

    # Gradient for a single batch
    function grad_fcn(G, u, args...)
        t.ps = u

        # Grid updating for likelihood model
        if  (t.st.train_idx == 1 || (t.st.train_idx - t.last_grid_update >= t.grid_update_frequency)) && (t.model.update_llhood_grid || t.model.update_prior_grid)
            t.model, t.ps, t.st, t.seed = update_model_grid(t.model, t.ps, Lux.testmode(t.st); seed=t.seed)
            t.grid_update_frequency = t.st.train_idx > 1 ? floor(t.grid_update_frequency * (2 - t.model.grid_update_decay)^t.st.train_idx) : t.grid_update_frequency
            t.last_grid_update = t.st.train_idx
            grid_updated = 1

            t.model.verbose && println("Iter: $(t.st.train_idx), Grid updated")
        end

        # Reduced precision grads, (switches to full precision for accumulation, not forward passes)
        result = CUDA.@fastmath withgradient(
            pars -> t.model.loss_fcn(
            t.model, 
            pars, 
            Lux.trainmode(t.st), 
            t.x; 
            seed=t.seed
            ), half_quant.(t.ps))
        t.loss, t.st, t.seed, grads = result.val..., first(result.grad) 
        t.loss = t.loss / t.model.loss_scaling
        grads = grads ./ t.model.loss_scaling
       
        isnan(norm(grads)) || isinf(norm(grads)) && find_nan(grads) 
        t.model.verbose && println("Iter: $(t.st.train_idx), Grad norm: $(norm(grads))")

        copy!(G, grads)
        return G
    end

    train_loss = 0

    # Train and test loss with logging
    function opt_loss(u, args...)
        t.ps = u
    
        train_loss += t.loss
        t.model.verbose && println("Iter: $(t.st.train_idx), Loss: $(t.loss)")

        # After one epoch, calculate test loss and log to CSV
        if t.st.train_idx % num_batches == 0 || t.st.train_idx == 1
            
            test_gen_loss = 0
            test_recon_loss = 0
            for (x, y) in t.test_loader
                x = reshape(x, 32*32, :) .|> half_quant |> device
                x_gen, t.st, t.seed = CUDA.@fastmath generate_batch(t.model, t.ps, Lux.testmode(t.st), size(x)[end]; seed=t.seed)
                x_rec, st_gen = t.model.lkhood.generate_from_z(t.model.lkhood, half_quant.(t.ps.gen), Lux.testmode(t.st.gen), x)
                @reset t.st.gen = st_gen
                x_rec, x_gen = x_rec .|> full_quant, x_gen .|> full_quant
                test_gen_loss += sum((device(y) - x_gen).^2) / size(x)[end]
                test_recon_loss += sum((device(y) - x_rec).^2) / size(x)[end]
            end
            
            train_loss = train_loss / num_batches
            test_gen_loss = test_gen_loss / length(t.test_loader)
            test_recon_loss = test_recon_loss / length(t.test_loader)

            now_time = time() - start_time
            epoch = t.st.train_idx == 1 ? 0 : fld(t.st.train_idx, num_batches)

            open(loss_file, "a") do file
                write(file, "$now_time,$(epoch),$train_loss,$test_gen_loss,$test_recon_loss,$grid_updated\n")
            end

            t.checkpoint && jldsave(t.model.file_loc * "ckpt_epoch_$(epoch).jld2"; params=t.ps |> cpu_device(), state=t.st |> cpu_device(), seed=t.seed)

            train_loss = 0
            grid_updated = 0
        end

        @reset t.st.train_idx += 1

        # Iterate loader, reset to first batch when epoch ends
        x, t.train_loader_state = (t.st.train_idx % num_batches == 0) ? iterate(t.model.train_loader) : iterate(t.model.train_loader, t.train_loader_state)
        t.x = device(x)
        return t.loss
    end    

    start_time = time()
    
    optf = Optimization.OptimizationFunction(opt_loss; grad=grad_fcn)
    optprob = Optimization.OptimizationProblem(optf, copy(t.ps))
    
    # Optimization only stops when maxiters is reached
    res = Optimization.solve(optprob, t.o.init_optimizer();
        maxiters=num_param_updates, 
        verbose=true,
        abstol=-full_quant(1),
        reltol=-full_quant(1),
        x_tol=-full_quant(1), 
        x_abstol=-full_quant(1), 
        x_reltol=-full_quant(1), 
        f_tol=-full_quant(1), 
        f_abstol=-full_quant(1), 
        f_reltol=-full_quant(1), 
        g_tol=-full_quant(1),
        g_abstol=-full_quant(1), 
        g_reltol=-full_quant(1),
        outer_x_abstol=-full_quant(1), 
        outer_x_reltol=-full_quant(1), 
        outer_f_abstol=-full_quant(1), 
        outer_f_reltol=-full_quant(1), 
        outer_g_abstol=-full_quant(1), 
        outer_g_reltol=-full_quant(1), 
        successive_f_tol=num_param_updates,
        allow_f_increases=true, 
        allow_outer_f_increases=true,
    )   

    t.ps = res.minimizer

    # Generate samples
    gen_data = zeros(half_quant, t.model.lkhood.x_shape..., 0) 
    idx = length(t.model.lkhood.x_shape) + 1
    for i in 1:(t.num_generated_samples // t.batch_size_for_gen)
        batch, t.st, t.seed = CUDA.@fastmath generate_batch(t.model, t.ps, Lux.testmode(t.st), t.batch_size_for_gen; seed=t.seed)
        gen_data = cat(gen_data, cpu_device()(batch), dims=idx)
    end

    recon_data = zeros(half_quant, t.model.lkhood.x_shape..., 0)
    perm_true = zeros(half_quant, t.model.lkhood.x_shape..., 0)
    flow_true = zeros(half_quant, t.model.lkhood.x_shape..., 0)
    for (x, y) in t.test_loader
        perm_true = cat(perm_true, cpu_device()(x), dims=idx)
        flow_true = cat(flow_true, cpu_device()(y), dims=idx)
        x = reshape(x, 32*32, :) .|> half_quant |> device
        x_rec, st_gen = t.model.lkhood.generate_from_z(t.model.lkhood, half_quant.(t.ps.gen), t.st.gen, x)
        @reset t.st.gen = st_gen
        x_rec, x = x_rec .|> full_quant, x .|> full_quant
        recon_data = cat(recon_data, cpu_device()(x_rec), dims=idx)
    end

    try
        h5write(t.model.file_loc * "generated_pressures.h5", "gen_samples", Float32.(gen_data))
        h5write(t.model.file_loc * "generated_pressures.h5", "recon_samples", Float32.(recon_data))
        h5write(t.model.file_loc * "generated_pressures.h5", "true_permeability", Float32.(perm_true))
        h5write(t.model.file_loc * "generated_pressures.h5", "true_flow", Float32.(flow_true))
    catch
        rm(t.model.file_loc * "generated_pressures.h5")
        h5write(t.model.file_loc * "generated_pressures.h5", "gen_samples", Float32.(gen_data))
        h5write(t.model.file_loc * "generated_pressures.h5", "recon_samples", Float32.(recon_data))
        h5write(t.model.file_loc * "generated_pressures.h5", "true_permeability", Float32.(perm_true))
        h5write(t.model.file_loc * "generated_pressures.h5", "true_flow", Float32.(flow_true))
    end

    t.save_model && jldsave(t.model.file_loc * "saved_model.jld2"; params=t.ps |> cpu_device(), state=t.st |> cpu_device())
end

end