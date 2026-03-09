using BenchmarkTools, ConfParser, Lux, Random, ComponentArrays, CSV, DataFrames, Reactant

ENV["DEVICE"] = "gpu"

include("../src/pipeline/data_utils.jl")
using .DataUtils: get_vision_dataset

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .KAEM_model

include("../src/KAEM/model_setup.jl")
using .ModelSetup

include("../src/pipeline/optimizer.jl")
using .optimization

conf = ConfParse("config/svhn_config.ini")
parse_conf!(conf)
optimizer = create_opt(conf)

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

function setup_model(n_z)
    commit!(conf, "EbmModel", "layer_widths", "$(n_z), $(2 * n_z + 1)")
    commit!(conf, "GeneratorModel", "widths", "$(2 * n_z + 1), $(4 * n_z + 2)")

    model = init_KAEM(dataset, conf, img_size; rng = rng)
    x, loader_state = iterate(model.train_loader)
    x = pu(x)
    model, _, params, st_kan, st_lux, st_rng = prep_model(model, x, optimizer; rng = rng, MLIR = false)

    return model, params, st_kan, st_lux, st_rng
end

function benchmark_generate(model, params, st_kan, st_lux, st_rng)
    return first(
        model(params, st_kan, Lux.testmode(st_lux), st_rng),
    )
end

results = DataFrame(
    n_z = Int[],
    time_mean = Float64[],
    time_std = Float64[],
    memory_estimate = Float64[],
    allocations = Int[],
    gc_percent = Float64[],
)

for n_z in [10, 20, 30, 40, 50]
    println("Benchmarking ITS generation (prior + decoder) with n_z = $n_z...")

    model, params, st_kan, st_lux, st_rng = setup_model(n_z)

    b = @benchmark begin
        result = f(
            $model,
            $params,
            $st_kan,
            $st_lux,
            $st_rng,
        )
        Reactant.synchronize(result)
    end setup = (
        f = Reactant.@compile sync = true benchmark_generate(
            $model,
            $params,
            $st_kan,
            $st_lux,
            $st_rng
        )
    )

    push!(
        results,
        (
            n_z,
            b.times[end] / 1.0e9,  # Convert to seconds (median time)
            std(b.times) / 1.0e9,  # Standard deviation
            b.memory / (1024^3),  # Convert to GiB
            b.allocs,
            b.gctimes[end] / b.times[end] * 100,  # Convert to percentage
        ),
    )
end

CSV.write("benches/results/ITS_generation.csv", results)
println("Results saved to ITS_generation.csv")
println(results)
