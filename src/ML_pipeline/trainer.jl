module trainer

export LV_KAM_trainer, init_trainer, train!

include("../LV-KAM/LV-KAM.jl")
include("optimizer.jl")
include("../utils.jl")
using .LV_KAM_model
using .optimization
using .Utils: device

using CUDA, KernelAbstractions, Tullio
using Random, MLDatasets, Images, ImageTransformations, ComponentArrays, CSV, HDF5, JLD2, ConfParser
using Zygote, Optimization, OptimizationOptimJL, Lux, LuxCUDA, LinearAlgebra

dataset_mapping = Dict(
    "MNIST" => MLDatasets.MNIST(),
    "FMNIST" => MLDatasets.FashionMNIST(),
    "CIFAR10" => MLDatasets.CIFAR10(),
    "SVHN" => MLDatasets.SVHN2(),
)

mutable struct LV_KAM_trainer
    model::LV_KAM
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
    iter::Int
    last_grid_update::Int
end

function init_trainer(rng::AbstractRNG, conf::ConfParse, dataset_name; 
    seed=1, img_resize=nothing, file_loc=nothing)

    # Load dataset
    N_train = parse(Int, retrieve(conf, "TRAINING", "N_train"))
    N_test = parse(Int, retrieve(conf, "TRAINING", "N_test"))
    dataset = dataset_mapping[dataset_name][1:N_train+N_test].features
    num_generated_samples = parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"))
    batch_size_for_gen = parse(Int, retrieve(conf, "TRAINING", "batch_size_for_gen"))

    # Option to resize dataset 
    dataset = isnothing(img_resize) ? dataset : imresize(dataset, img_resize)
    img_shape = size(dataset)[1:end-1]
    dataset = reshape(dataset, prod(size(dataset)[1:end-1]), size(dataset)[end]) .|> Float32
    save_dataset = reshape(dataset[:, 1:num_generated_samples], img_shape..., num_generated_samples)
    println("Resized dataset to $(img_shape)")
    
    # Initialize model
    model = init_LV_KAM(dataset, conf; prior_seed=seed, lkhood_seed=seed, data_seed=seed)
    params, state = Lux.setup(rng, model)
    optimizer = create_opt(conf)
    grid_update_frequency = parse(Int, retrieve(conf, "MOE_LIKELIHOOD", "grid_update_frequency"))

    N_epochs = parse(Int, retrieve(conf, "TRAINING", "N_epochs"))
    x, loader_state = iterate(model.train_loader) 

    N_t = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps"))
    model_type = N_t > 1 ? "Thermodynamic" : "Vanilla"

    file_loc = isnothing(file_loc) ? "logs/$(model_type)/$(dataset_name)_$(seed)/" : file_loc
    mkpath(file_loc)

    try
        h5write(file_loc * "real_images.h5", "samples", save_dataset)
    catch
        rm(file_loc * "real_images.h5")
        h5write(file_loc * "real_images.h5", "samples", save_dataset)
    end
    
    return LV_KAM_trainer(
        model, 
        optimizer, 
        dataset_name, 
        img_shape, 
        ComponentArray(params) |> device, 
        device(state), 
        N_epochs, 
        loader_state, 
        device(x'), 
        file_loc, 
        num_generated_samples,
        batch_size_for_gen,
        seed, 
        grid_update_frequency,
        1, 
        1)
end

function train!(t::LV_KAM_trainer)
    num_batches = length(t.model.train_loader)
    grid_updated = 0
    num_param_updates = num_batches * t.N_epochs
    
    loss_file = t.file_loc * "loss.csv"
    open(loss_file, "w") do file
        write(file, "Time (s),Epoch,Train Loss,Test Loss,Grid Updated\n")
    end

    function find_nan(grads)
        for k in keys(grads)
            if any(isnan, grads[k]) || any(isinf, grads[k])
                for i in keys(grads[k])
                    any(isnan, grads[k][i]) || any(isinf, grads[k][i]) && error("NaN/Inf in $k, $i gradients")
                end
            end
        end
    end

    # Gradient for a single batch
    function grad_fcn(G, u, args...)
        t.ps = u

        # Grid updating for likelihood model
        if  (t.iter == 1 || (t.iter - t.last_grid_update >= t.grid_update_frequency))
            t.model, t.ps, t.seed = update_llhood_grid(t.model, t.ps, t.st; seed=t.seed)
            t.grid_update_frequency = t.iter > 1 ? floor(t.grid_update_frequency * (2 - t.model.grid_update_decay)^t.iter) : t.grid_update_frequency
            t.last_grid_update = t.iter
            grid_updated = 1

            t.model.verbose && println("Iter: $(t.iter), Grid updated")
        end

        grads = first(gradient(pars -> MLE_loss(t.model, pars, t.st, t.x; seed=t.seed), t.ps))
        any(isnan, grads) ||any(isinf, grads) && find_nan(grads)
        t.seed += 1

        t.model.verbose && println("Iter: $(t.iter), Grad norm: $(norm(grads))")

        copy!(G, grads)
        return G
    end

    train_loss = 0

    # Train and test loss with logging
    function opt_loss(u, args...)
        t.ps = u
        loss = MLE_loss(t.model, t.ps, t.st, t.x)
        train_loss += loss
        t.model.verbose && println("Iter: $(t.iter), Loss: $loss")

        # After one epoch, calculate test loss and log to CSV
        if t.iter % num_batches == 0 || t.iter == 1
            
            test_loss = 0
            for x in t.model.test_loader
                x = device(x')
                test_loss += MLE_loss(t.model, t.ps, t.st, x; seed=t.seed)
                t.seed += 1
            end
            
            train_loss = train_loss / num_batches
            test_loss /= length(t.model.test_loader)
            now_time = time() - start_time
            epoch = t.iter == 1 ? 0 : fld(t.iter, num_batches)

            open(loss_file, "a") do file
                write(file, "$now_time,$(epoch),$train_loss,$test_loss,$grid_updated\n")
            end

            train_loss = 0
            grid_updated = 0
        end

        t.iter += 1

        # Iterate loader, reset to first batch when epoch ends
        x, t.train_loader_state = (t.iter % num_batches == 0) ? iterate(t.model.train_loader) : iterate(t.model.train_loader, t.train_loader_state)
        t.x = device(x')

        return loss
    end    

    start_time = time()
    
    optf = Optimization.OptimizationFunction(opt_loss; grad=grad_fcn)
    optprob = Optimization.OptimizationProblem(optf, copy(t.ps))
    
    # Optimization only stops when maxiters is reached
    res = Optimization.solve(optprob, t.o.init_optimizer();
        maxiters=num_param_updates, 
        verbose=true,
        abstol=-1f0,
        reltol=-1f0,
        x_tol=-1f0, 
        x_abstol=-1f0, 
        x_reltol=-1f0, 
        f_tol=-1f0, 
        f_abstol=-1f0, 
        f_reltol=-1f0, 
        g_tol=-1f0,
        g_abstol=-1f0, 
        g_reltol=-1f0,
        outer_x_abstol=-1f0, 
        outer_x_reltol=-1f0, 
        outer_f_abstol=-1f0, 
        outer_f_reltol=-1f0, 
        outer_g_abstol=-1f0, 
        outer_g_reltol=-1f0, 
        successive_f_tol=num_param_updates,
        allow_f_increases=true, 
        allow_outer_f_increases=true,
    )   

    t.ps = res.minimizer

    # Generate samples
    generated_images = zeros(Float32, 0, t.img_shape...) 
    for i in 1:(t.num_generated_samples // t.batch_size_for_gen)
        batch, t.seed = generate_batch(t.model, t.ps, t.st, t.batch_size_for_gen; seed=t.seed)
        batch = cpu_device()(reshape(batch, t.batch_size_for_gen, t.img_shape...))
        generated_images = vcat(generated_images, batch)
        t.seed += 1
    end

    try
        h5write(t.file_loc * "generated_images.h5", "samples", generated_images)
    catch
        rm(t.file_loc * "generated_images.h5")
        h5write(t.file_loc * "generated_images.h5", "samples", generated_images)
    end

    # Save params, state, model
    jldsave(t.file_loc * "saved_model.jld2"; params=cpu_device()(t.ps), state=cpu_device()(t.st), model=t.model)
end

end