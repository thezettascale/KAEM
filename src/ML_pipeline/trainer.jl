module trainer

export T_KAM_trainer, init_trainer, train!

include("../T-KAM/T-KAM.jl")
include("optimizer.jl")
include("../utils.jl")
using .T_KAM_model
using .optimization
using .Utils: device, half_quant, full_quant, move_to_cpu

using CUDA, KernelAbstractions, Tullio
using Random, MLDatasets, Images, ImageTransformations, ComponentArrays, CSV, HDF5, JLD2, ConfParser
using Zygote, Optimization, OptimizationOptimJL, Lux, LuxCUDA, LinearAlgebra, Accessors

dataset_mapping = Dict(
    "MNIST" => MLDatasets.MNIST(),
    "FMNIST" => MLDatasets.FashionMNIST(),
    "CIFAR10" => MLDatasets.CIFAR10(),
    "SVHN" => MLDatasets.SVHN2(),
)

mutable struct T_KAM_trainer
    model
    o::opt
    dataset_name::AbstractString
    img_shape::Tuple
    ps::ComponentArray
    st::NamedTuple
    N_epochs::Int
    train_loader_state::Tuple{Any, Int}
    x::AbstractArray
    file_loc::AbstractString
    num_generated_samples::Int
    batch_size_for_gen::Int
    seed::Int
    grid_update_frequency::Int
    last_grid_update::Int
    save_model::Bool
end

function init_trainer(rng::AbstractRNG, conf::ConfParse, dataset_name; 
    seed=1, img_resize=nothing, file_loc=nothing, save_model=true)

    # Load dataset
    N_train = parse(Int, retrieve(conf, "TRAINING", "N_train"))
    N_test = parse(Int, retrieve(conf, "TRAINING", "N_test"))
    dataset = dataset_mapping[dataset_name][1:N_train+N_test].features
    num_generated_samples = parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"))
    batch_size_for_gen = parse(Int, retrieve(conf, "TRAINING", "batch_size_for_gen"))
    cnn = dataset_name == "CIFAR10" || dataset_name == "SVHN" 
    commit!(conf, "CNN", "use_cnn_lkhood", string(cnn))

    # Option to resize dataset 
    dataset = isnothing(img_resize) ? dataset : imresize(dataset, img_resize)
    img_shape = size(dataset)[1:end-1]
    dataset = cnn ? dataset : reshape(dataset, prod(size(dataset)[1:end-1]), size(dataset)[end])
    dataset = dataset .|> half_quant
    save_dataset = cnn ? dataset[:,:,:,1:num_generated_samples] : reshape(dataset[:, 1:num_generated_samples], img_shape..., num_generated_samples)
    println("Resized dataset to $(img_shape)")
    
    # Initialize model
    model = init_T_KAM(dataset, conf; prior_seed=seed, lkhood_seed=seed, data_seed=seed)
    params, state = Lux.setup(rng, model)
    optimizer = create_opt(conf)
    grid_update_frequency = parse(Int, retrieve(conf, "GRID_UPDATING", "grid_update_frequency"))

    N_epochs = parse(Int, retrieve(conf, "TRAINING", "N_epochs"))
    x, loader_state = iterate(model.train_loader) 

    N_t = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps"))
    mala = parse(Bool, retrieve(conf, "MALA", "use_langevin")) ? "MALA" : "importance"
    model_type = N_t > 1 ? "Thermodynamic" : "Vanilla/$mala"

    file_loc = isnothing(file_loc) ? "logs/$(model_type)/$(dataset_name)_$(seed)/" : file_loc
    mkpath(file_loc)

    try
        h5write(file_loc * "real_images.h5", "samples", Float32.(save_dataset))
    catch
        rm(file_loc * "real_images.h5")
        h5write(file_loc * "real_images.h5", "samples", Float32.(save_dataset))
    end
    
    return T_KAM_trainer(
        model, 
        optimizer, 
        dataset_name, 
        img_shape, 
        device(params), 
        device(state), 
        N_epochs, 
        loader_state, 
        device(x), 
        file_loc, 
        num_generated_samples,
        batch_size_for_gen,
        seed, 
        grid_update_frequency,
        1,
        save_model
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
    
    loss_file = t.file_loc * "loss.csv"
    open(loss_file, "w") do file
        write(file, "Time (s),Epoch,Train MLE Loss,Test MSE Loss,Grid Updated\n")
    end

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
            t.model, t.ps, t.seed = update_model_grid(t.model, t.ps, t.st; seed=t.seed)
            t.grid_update_frequency = t.st.train_idx > 1 ? floor(t.grid_update_frequency * (2 - t.model.grid_update_decay)^t.st.train_idx) : t.grid_update_frequency
            t.last_grid_update = t.st.train_idx
            grid_updated = 1

            t.model.verbose && println("Iter: $(t.st.train_idx), Grid updated")
        end

        # Reduced precision grads
        grads = CUDA.@fastmath first(gradient(pars -> first(t.model.loss_fcn(
            t.model, 
            pars, 
            t.st, 
            t.x; 
            full_precision=false,
            seed=t.seed
            )), 
            half_quant.(t.ps))) .|> full_quant

        grads = grads ./ loss_scaling
        
        isnan(norm(grads)) || isinf(norm(grads)) && find_nan(grads) 
        t.model.verbose && println("Iter: $(t.st.train_idx), Grad norm: $(norm(grads))")

        copy!(G, grads)
        return G
    end

    train_loss = 0

    # Train and test loss with logging
    function opt_loss(u, args...)
        t.ps = u
        
        # Full precision loss, (switches to full precision for accumulation, not forward passes)
        loss, t.st, t.seed = CUDA.@fastmath t.model.loss_fcn(
            t.model, 
            half_quant.(t.ps), 
            Lux.testmode(t.st), 
            t.x; 
            full_precision=true, 
            seed=t.seed
            )

        loss = loss / loss_scaling
        
        train_loss += loss
        t.model.verbose && println("Iter: $(t.st.train_idx), Loss: $loss")

        # After one epoch, calculate test loss and log to CSV
        if t.st.train_idx % num_batches == 0 || t.st.train_idx == 1
            
            test_loss = 0
            for x in t.model.test_loader
                x_gen, t.seed = generate_batch(t.model, t.ps, t.st, size(x, 2); seed=t.seed)
                test_loss += sum((x_gen' .- device(x)).^2) / size(x, 2)
            end
            
            train_loss = train_loss / num_batches
            test_loss /= length(t.model.test_loader)
            now_time = time() - start_time
            epoch = t.st.train_idx == 1 ? 0 : fld(t.st.train_idx, num_batches)

            open(loss_file, "a") do file
                write(file, "$now_time,$(epoch),$train_loss,$test_loss,$grid_updated\n")
            end

            train_loss = 0
            grid_updated = 0
        end

        @reset t.st.train_idx += 1

        # Iterate loader, reset to first batch when epoch ends
        x, t.train_loader_state = (t.st.train_idx % num_batches == 0) ? iterate(t.model.train_loader) : iterate(t.model.train_loader, t.train_loader_state)
        t.x = device(x)
        return loss
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
    generated_images = zeros(half_quant, 0, t.img_shape...) 
    for i in 1:(t.num_generated_samples // t.batch_size_for_gen)
        batch, t.seed = generate_batch(t.model, t.ps, t.st, t.batch_size_for_gen; seed=t.seed)
        batch = cpu_device()(reshape(batch, t.batch_size_for_gen, t.img_shape...))
        generated_images = vcat(generated_images, batch)
    end

    try
        h5write(t.file_loc * "generated_images.h5", "samples", Float32.(generated_images))
    catch
        rm(t.file_loc * "generated_images.h5")
        h5write(t.file_loc * "generated_images.h5", "samples", Float32.(generated_images))
    end

    # Save params, state, model
    if t.save_model
        model, ps, st = move_to_cpu(t.model, t.ps, t.st)
        jldsave(t.file_loc * "saved_model.jld2"; params=ps, state=st, model=model)
    end
end

end