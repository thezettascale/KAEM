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

function setup_model(N_t)
    commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "$(N_t)")

    model = init_KAEM(dataset, conf, img_size; rng = rng)
    x_test, loader_state = iterate(model.train_loader)
    x_test = pu(x_test)
    model, opt_state, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; rng = rng, MLIR = false)
    swap_replica_idxs = rand(1:(model.N_t - 1), model.posterior_sampler.N)

    return model, opt_state, ps, st_kan, st_lux, st_rng, x_test, swap_replica_idxs
end

results = DataFrame(
    N_t = Int[],
    time_mean = Float64[],
    time_std = Float64[],
    memory_estimate = Float64[],
    allocations = Int[],
    gc_percent = Float64[],
)

function benchmark_temps(
        opt_state,
        params,
        st_kan,
        st_lux,
        st_rng,
        model,
        x_test,
        swap
    )
    return model.train_step(
        opt_state,
        params,
        st_kan,
        st_lux,
        x_test,
        1,
        st_rng,
    )
end

for N_t in [2, 4, 6, 8, 10, 12]
    println("Benchmarking N_t = $N_t...")

    model, opt_state, ps, st_kan, st_lux, st_rng, x_test, swap = setup_model(N_t)

    b = @benchmark begin
        result = f(
            $opt_state,
            $ps,
            $st_kan,
            $st_lux,
            $st_rng,
            $model,
            $x_test,
            $swap
        )
        Reactant.synchronize(result)
    end setup = (
        f = Reactant.@compile sync = true benchmark_temps(
            $opt_state,
            $ps,
            $st_kan,
            $st_lux,
            $st_rng,
            $model,
            $x_test,
            $swap
        )
    )

    push!(
        results,
        (
            N_t,
            b.times[end] / 1.0e9,  # Convert to seconds (median time)
            std(b.times) / 1.0e9,  # Standard deviation
            b.memory / (1024^3),  # Convert to GiB
            b.allocs,
            b.gctimes[end] / b.times[end] * 100,  # Convert to percentage
        ),
    )
end

CSV.write("benches/results/temperatures.csv", results)
println("Results saved to temperatures.csv")
println(results)
