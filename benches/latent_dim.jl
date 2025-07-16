using BenchmarkTools, ConfParser, Lux, Random, CUDA, ComponentArrays, CSV, DataFrames

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/T-KAM/T-KAM.jl")
include("../src/pipeline/data_utils.jl")
include("../src/utils.jl")
using .T_KAM_model
using .DataUtils: get_vision_dataset
using .Utils: device, half_quant

conf = ConfParse("config/nist_config.ini")
parse_conf!(conf)
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

    model = init_T_KAM(dataset, conf, img_size)
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    model = move_to_hq(model)
    x_test = device(first(model.train_loader))
    ps, st = ComponentArray(ps) |> device, st |> device
    ∇ = zero(half_quant.(ps))

    return model, half_quant.(ps), ∇, st, x_test
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
    println("Benchmarking n_z = $n_z...")

    model, ps, ∇, st, x_test = setup_model(n_z)

    CUDA.reclaim()
    GC.gc()

    b = @benchmark model.loss_fcn($ps, $∇, $st, $model, $x_test)

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
