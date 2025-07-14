using BenchmarkTools,
    ConfParser,
    Lux,
    DifferentiationInterface,
    Random,
    CUDA,
    ComponentArrays,
    CSV,
    DataFrames

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"
ENV["AD_BACKEND"] = "ENZYME"

include("../src/T-KAM/T-KAM.jl")
include("../src/pipeline/data_utils.jl")
include("../src/utils.jl")
using .T_KAM_model
using .DataUtils: get_vision_dataset
using .Utils: device, half_quant, AD_backend

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
    cnn = true,
)[1:2]

function setup_model(N_t)
    commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "$(N_t)")

    model = init_T_KAM(dataset, conf, img_size)
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    model = move_to_hq(model)
    x_test = device(first(model.train_loader))
    ps, st = ComponentArray(ps) |> device, st |> device

    return model, ps, st, x_test
end

function benchmark_temp(model, ps, st, x_test)
    first(
        gradient(
            p -> first(model.loss_fcn(p, st, model, x_test)),
            AD_backend,
            half_quant.(ps),
        ),
    )
end

results = DataFrame(
    N_t = Int[],
    time_mean = Float64[],
    time_std = Float64[],
    memory_estimate = Float64[],
    allocations = Int[],
    gc_percent = Float64[],
)

for N_t in [1, 2, 4, 6, 8, 10]
    println("Benchmarking N_t = $N_t...")

    model, ps, st, x_test = setup_model(N_t)

    CUDA.reclaim()
    GC.gc()

    b = @benchmark benchmark_temp($model, $ps, $st, $x_test)

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

CSV.write("benches/results/temperatures.csv", results)
println("Results saved to temperatures.csv")
println(results)
