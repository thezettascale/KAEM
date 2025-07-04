using BenchmarkTools, ConfParser, Lux, Zygote, Random, CUDA, ComponentArrays, CSV, DataFrames

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/T-KAM/T-KAM.jl")
include("../src/ML_pipeline/data_utils.jl")
include("../src/utils.jl")
using .T_KAM_model
using .DataUtils: get_vision_dataset
using .Utils: device, half_quant

conf = ConfParse("benches/svhn_prior_config.ini")
parse_conf!(conf)

commit!(conf, "CNN", "use_cnn_lkhood", "true")
commit!(conf, "SEQ", "sequence_length", "0") 
commit!(conf, "TRAINING", "verbose", "false") 
commit!(conf, "POST_LANGEVIN", "use_langevin", "true")
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "1")

dataset, img_size = get_vision_dataset(
    "SVHN",
    parse(Int, retrieve(conf, "TRAINING", "N_train")),
    parse(Int, retrieve(conf, "TRAINING", "N_test")),
    parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"));
    cnn=true
)[1:2]

function benchmark_prior(N_l)
    CUDA.reclaim()  
    GC.gc()        
    
    commit!(conf, "PRIOR_LANGEVIN", "iters", "$(N_l)")
    model = init_T_KAM(dataset, conf, img_size)
    

    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    model = move_to_hq(model)
    x_test = device(first(model.train_loader))
    ps, st = ComponentArray(ps) |> device, st |> device 

    first(gradient(p -> first(model.loss_fcn(model, p, st, x_test)), half_quant.(ps)))
end


results = DataFrame(N_l=Int[], time_mean=Float64[], time_std=Float64[], memory_estimate=Float64[], allocations=Int[], gc_percent=Float64[])

for N_l in [10, 20, 30, 40, 50]
    println("Benchmarking N_l = $N_l...")
    b = @benchmark benchmark_prior(data) setup=(data=$N_l)
    
    push!(results, (
        N_l,
        b.times[end] / 1e9,  # Convert nanoseconds to seconds (median time)
        std(b.times) / 1e9,  # Standard deviation
        b.memory / (1024^3),  # Convert bytes to GiB
        b.allocs,
        b.gctimes[end] / b.times[end] * 100  # GC percentage
    ))
end

# Save results to CSV
CSV.write("benches/results/prior_steps.csv", results)
println("Results saved to prior_steps.csv")
println(results)
