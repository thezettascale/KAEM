using BenchmarkTools, ConfParser, Lux, Random, CUDA, ComponentArrays, CSV, DataFrames

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/pipeline/data_utils.jl")
using .DataUtils: get_vision_dataset

include("../src/utils.jl")
using .Utils

include("../src/T-KAM/T-KAM.jl")
using .T_KAM_model

include("../src/T-KAM/model_setup.jl")
using .ModelSetup

conf = ConfParse("config/svhn_config.ini")
parse_conf!(conf)

rng = Random.MersenneTwister(1)

commit!(conf, "CNN", "use_cnn_lkhood", "true")
commit!(conf, "SEQ", "sequence_length", "0")
commit!(conf, "TRAINING", "verbose", "false")
commit!(conf, "POST_LANGEVIN", "use_langevin", "true")

dataset, img_size = get_vision_dataset(
    "SVHN",
    parse(Int, retrieve(conf, "TRAINING", "N_train")),
    parse(Int, retrieve(conf, "TRAINING", "N_test")),
    parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"));
    cnn = true,
)[1:2]

function setup_model(N_t)
    commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "$(N_t)")

    model = init_T_KAM(dataset, conf, img_size; rng = rng)
    x, loader_state = iterate(model.train_loader)
    x = pu(x)
    model, params, st_kan, st_lux = prep_model(model, x; rng = rng)

    return model, half_quant.(params), st_kan, st_lux
end

function benchmark_prior(model, params, st_kan, st_lux)
    first(model.sample_prior(model, model.grid_updates_samples, params, st_kan, st_lux, rng))
end

results = DataFrame(
    N_t = Int[],
    time_mean = Float64[],
    time_std = Float64[],
    memory_estimate = Float64[],
    allocations = Int[],
    gc_percent = Float64[],
)

for N_t in [1]
    println("Benchmarking N_t = $N_t...")

    model, params, st_kan, st_lux = setup_model(N_t)

    CUDA.reclaim()
    GC.gc()

    b = @benchmark benchmark_prior($model, $params, $st_kan, $st_lux)

    push!(
        results,
        (
            N_t,
            b.times[end] / 1e9,  # Convert to seconds (median time)
            std(b.times) / 1e9,  # Standard deviation
            b.memory / (1024^3),  # Convert to GiB
            b.allocs,
            b.gctimes[end] / b.times[end] * 100,  # Convert to percentage
        ),
    )
end

CSV.write("benches/results/ITS_single.csv", results)
println("Results saved to ITS_single.csv")
println(results)
