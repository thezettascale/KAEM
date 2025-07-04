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

conf = ConfParse("config/svhn_config.ini")
parse_conf!(conf)

commit!(conf, "CNN", "use_cnn_lkhood", "true")
commit!(conf, "SEQ", "sequence_length", "0") 
commit!(conf, "TRAINING", "verbose", "false") 
commit!(conf, "POST_LANGEVIN", "use_langevin", "true")

dataset, img_size = get_vision_dataset(
    "SVHN",
    parse(Int, retrieve(conf, "TRAINING", "N_train")),
    parse(Int, retrieve(conf, "TRAINING", "N_test")),
    parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"));
    cnn=true
)[1:2]

function benchmark_MALA(N_l)
    CUDA.reclaim()  
    GC.gc()        
    
    commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "$(N_l)")

    model = init_T_KAM(dataset, conf, img_size)
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    model = move_to_hq(model)
    x_test = device(first(model.train_loader))
    ps, st = ComponentArray(ps) |> device, st |> device 

    first(gradient(p -> first(model.loss_fcn(model, p, st, x_test)), half_quant.(ps)))
end

results = DataFrame(N_t=Int[], time_mean=Float64[], time_std=Float64[], memory_estimate=Float64[], allocations=Int[], gc_percent=Float64[])

for N_t in [2, 4, 6, 8, 10, 12]
    println("Benchmarking N_t = $N_t...")
    b = @benchmark benchmark_MALA(data) setup=(data=$N_t)
    
    push!(results, (
        N_t,
        b.times[end] / 1e9,  # Convert to seconds (median time)
        std(b.times) / 1e9,  # Standard deviation
        b.memory / (1024^3),  # Convert to GiB
        b.allocs,
        b.gctimes[end] / b.times[end] * 100  # Convert to percentage
    ))
end

CSV.write("benches/results/temperatures.csv", results)
println("Results saved to temperatures.csv")
println(results)
