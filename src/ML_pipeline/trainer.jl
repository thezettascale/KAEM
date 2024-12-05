module trainer

export LV_KAM_trainer, init_trainer, train!

include("../LV-KAM/LV-KAM.jl")
include("optimizer.jl")
include("../utils.jl")
using .LV_KAM_model
using .optimization
using .Utils: device

using CUDA, KernelAbstractions, Tullio
using Random, MLDatasets, Images, ImageTransformations, ComponentArrays, CSV, HDF5, BSON, ConfParser
using Zygote, Optimization, OptimizationOptimJL, Lux, LuxCUDA

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
    println("Resized dataset to $(img_shape)")
    
    # Initialize model
    model = init_LV_KAM(dataset, conf; prior_seed=seed, lkhood_seed=seed, data_seed=seed)
    params, state = Lux.setup(rng, model)
    optimizer = create_opt(conf)
    grid_update_frequency = parse(Int, retrieve(conf, "MOE_LIKELIHOOD", "grid_update_frequency"))

    N_epochs = parse(Int, retrieve(conf, "TRAINING", "N_epochs"))
    x, loader_state = iterate(model.train_loader) 

    file_loc = isnothing(file_loc) ? "logs/$(dataset_name)_$(seed)/" : file_loc
    mkpath(file_loc)
    
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
        write(file, "Time (s),Iter,Batch Loss,Test Loss,Grid Updated\n")
    end

    function find_nan(grads)
        for k in keys(grads)
            if any(isnan.(grads[k]))
                for i in keys(grads[k])
                    any(isnan.(grads[k][i])) && error("NaN in $k, $i gradients")
                end
            end
        end
    end

    # Train step for a single batch
    function grad_fcn(G, u, args...)
        t.ps = u
        grid_updated = 0

        # Grid updating for likelihood model
        if  (t.iter == 1 || (t.iter - t.last_grid_update >= t.grid_update_frequency))
            t.model, t.ps, t.seed = update_llhood_grid(t.model, t.ps, t.st; seed=t.seed)
            t.grid_update_frequency = t.iter > 1 ? floor(t.grid_update_frequency * (2 - t.model.grid_update_decay)^t.iter) : t.grid_update_frequency
            t.last_grid_update = t.iter
            grid_updated = 1
        end

        grads = first(gradient(pars -> MLE_loss(t.model, pars, t.st, t.x; seed=t.seed), t.ps))
        any(isnan, grads) && find_nan(grads)
        t.seed += 1

        copy!(G, grads)
        return G
    end

    function opt_loss(u, args...)
        t.ps = u
        return MLE_loss(t.model, t.ps, t.st, t.x)
    end    

    start_time = time()

    # Callback for logging
    function log_callback!(state, obj)
        t.ps = state.u
        
        # After one epoch only
        if t.iter % num_batches == 0 || t.iter == 1
            
            test_loss = 0
            for x in t.model.test_loader
                x = device(x')
                test_loss += MLE_loss(t.model, t.ps, t.st, x; seed=t.seed)
                t.seed += 1
            end
            
            test_loss /= length(t.model.test_loader)
            now_time = time() - start_time

            open(loss_file, "a") do file
                write(file, "$now_time,$t.iter,$obj,$test_loss,$grid_updated\n")
            end
        end

        t.iter += 1

        # Iterate loader, reset to first batch when epoch ends
        x, t.train_loader_state = (t.iter % num_batches == 0) ? iterate(t.model.train_loader) : iterate(t.model.train_loader, t.train_loader_state)
        t.x = device(x')

        return false
    end
    
    optf = Optimization.OptimizationFunction(opt_loss; grad=grad_fcn)
    optprob = Optimization.OptimizationProblem(optf, copy(t.ps), nothing)
    
    # Optimization only stops when maxiters is reached
    res = Optimization.solve(optprob, t.o.init_optimizer();
        maxiters=num_param_updates, 
        cb=log_callback!, 
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
        generated_images = vcat(generated_images, cpu_device()(batch))
        t.seed += 1
    end
    h5write(t.file_loc * "generated_images.h5", "samples", generated_images)

    # Save params, state, model
    BSON.bson(t.file_loc * "params.bson", cpu_device()(t.ps))  
    BSON.bson(t.file_loc * "state.bson", cpu_device()(t.st))  
    BSON.bson(t.file_loc * "model.bson", t.model)  
end

end