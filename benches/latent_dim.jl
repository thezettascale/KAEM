using BenchmarkTools, ConfParser, Lux, Random, CUDA, ComponentArrays, CSV, DataFrames

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/pipeline/data_utils.jl")
using .DataUtils: get_vision_dataset

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .T_KAM_model

include("../src/KAEM/model_setup.jl")
using .ModelSetup

conf = ConfParse("config/nist_config.ini")
parse_conf!(conf)

rng = Random.MersenneTwister(1)

commit!(conf, "POST_LANGEVIN", "use_langevin", "false")
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
commit!(conf, "CNN", "use_cnn_lkhood", "false")
commit!(conf, "SEQ", "sequence_length", "0")
commit!(conf, "TRAINING", "verbose", "false")

dataset, img_size = get_vision_dataset(
    "MNIST",
    parse(Int, retrieve(conf, "TRAINING", "N_train")),
    parse(Int, retrieve(conf, "TRAINING", "N_test")),
    parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"));
)[1:2]

function setup_model(n_z)
    commit!(conf, "EbmModel", "layer_widths", "$(n_z), $(2*n_z+1)")
    commit!(conf, "GeneratorModel", "widths", "$(2*n_z+1), $(4*n_z+2)")

    model = init_T_KAM(dataset, conf, img_size; rng = rng)
    x_test, loader_state = iterate(model.train_loader)
    x_test = pu(x_test)
    model, params, st_kan, st_lux = prep_model(model, x_test; rng = rng)
    ∇ = zero(half_quant.(params))

    return model, half_quant.(params), ∇, st_kan, st_lux, x_test
end

results = DataFrame(
    n_z = Int[],
    time_mean = Float64[],
    time_std = Float64[],
    memory_estimate = Float64[],
    allocations = Int[],
    gc_percent = Float64[],
)

function benchmark_latent_dim(params, ∇, st_kan, st_lux, model, x_test)
    model.loss_fcn(params, ∇, st_kan, st_lux, model, x_test)
end

for n_z in [10, 20, 30, 40, 50]
    println("Benchmarking n_z = $n_z...")

    model, params, ∇, st_kan, st_lux, x_test = setup_model(n_z)

    CUDA.reclaim()
    GC.gc()

    b = @benchmark benchmark_latent_dim($params, $∇, $st_kan, $st_lux, $model, $x_test)

    push!(
        results,
        (
            n_z,
            b.times[end] / 1e9,  # Convert to seconds (median time)
            std(b.times) / 1e9,  # Standard deviation
            b.memory / (1024^3),  # Convert to GiB
            b.allocs,
            b.gctimes[end] / b.times[end] * 100,  # GC percentage
        ),
    )
end

CSV.write("benches/results/latent_dim.csv", results)
println("Results saved to latent_dim.csv")
println(results)
