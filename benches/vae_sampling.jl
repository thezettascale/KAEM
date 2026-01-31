using BenchmarkTools, ConfParser, Lux, Random, ComponentArrays, CSV, DataFrames, Reactant

ENV["GPU"] = true

include("../src/baseline/training/trainer.jl")
using .Baseline: Utils, VAEModel, DataUtils
using .Utils: pu
using .VAEModel: init_VAE, sample
using .DataUtils: get_vision_dataset

conf = ConfParse("config/baseline_svhn_config.ini")
parse_conf!(conf)

rng = Random.MersenneTwister(1)

commit!(conf, "TRAINING", "verbose", "false")

dataset, img_size = get_vision_dataset(
    "SVHN",
    parse(Int, retrieve(conf, "TRAINING", "N_train")),
    parse(Int, retrieve(conf, "TRAINING", "N_test")),
    parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"));
    cnn = true,
)[1:2]

function setup_vae_model(latent_dim)
    commit!(conf, "VAE", "latent_dim", "$(latent_dim)")

    model = init_VAE(conf, img_size; rng = rng)

    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)
    ps, st = ps |> ComponentArray |> Lux.f32 |> pu, st |> Lux.f32 |> pu

    z = randn(rng, Float32, model.latent_dim, model.batch_size) |> pu

    return model, ps, st, z
end

function benchmark_vae_sample(model, ps, st, z)
    return first(
        sample(model, ps, Lux.testmode(st), z),
    )
end

results = DataFrame(
    latent_dim = Int[],
    time_mean = Float64[],
    time_std = Float64[],
    memory_estimate = Float64[],
    allocations = Int[],
    gc_percent = Float64[],
)

# VAE latent dims correspond to (2n+1) where n is the KAEM latent dim
# For KAEM n_z = [10, 20, 30, 40, 50], VAE latent_dim = [21, 41, 61, 81, 101]
for latent_dim in [21, 41, 61, 81, 101]
    println("Benchmarking VAE sampling with latent_dim = $latent_dim...")

    model, ps, st, z = setup_vae_model(latent_dim)

    b = @benchmark begin
        result = f(
            $model,
            $ps,
            $st,
            $z,
        )
        Reactant.synchronize(result)
    end setup = (
        f = Reactant.@compile sync = true benchmark_vae_sample(
            $model,
            $ps,
            $st,
            $z
        )
    )

    push!(
        results,
        (
            latent_dim,
            b.times[end] / 1.0e9,  # Convert to seconds
            std(b.times) / 1.0e9,  # Standard deviation
            b.memory / (1024^3),  # Convert to GiB
            b.allocs,
            b.gctimes[end] / b.times[end] * 100,  # GC percentage
        ),
    )
end

CSV.write("benches/results/vae_sampling.csv", results)
println("Results saved to vae_sampling.csv")
println(results)
