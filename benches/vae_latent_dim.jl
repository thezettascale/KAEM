using BenchmarkTools, ConfParser, Lux, Random, ComponentArrays, CSV, DataFrames, Reactant, Optimisers, Flux

ENV["GPU"] = true

include("../src/baseline/training/trainer.jl")
using .Baseline: Utils, VAEModel, VAELoss, BaselineRNG, optimization, DataUtils
using .Utils: pu
using .VAEModel: init_VAE
using .VAELoss: VAETrainStep
using .BaselineRNG: seed_rng
using .optimization: ManualAdam
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

    lr_vae = parse(Float32, retrieve(conf, "VAE", "learning_rate"))
    opt_vae = ManualAdam(lr_vae)
    opt_state = Optimisers.setup(opt_vae, ps)

    st_rng = seed_rng(model; rng = rng)
    β = parse(Float32, retrieve(conf, "VAE", "beta"))

    train_loader = Flux.DataLoader(dataset; batchsize = model.batch_size, shuffle = true)
    x_test, _ = iterate(train_loader)
    x_test = pu(x_test)

    return model, opt_state, ps, st, st_rng, β, x_test
end

results = DataFrame(
    latent_dim = Int[],
    time_mean = Float64[],
    time_std = Float64[],
    memory_estimate = Float64[],
    allocations = Int[],
    gc_percent = Float64[],
)

function benchmark_vae_train(train_step, opt_state, ps, st, x, st_rng)
    return train_step(opt_state, ps, Lux.trainmode(st), x, st_rng)
end

# VAE latent dims correspond to (2n+1) where n is the KAEM latent dim
# For KAEM n_z = [10, 20, 30, 40, 50], VAE latent_dim = [21, 41, 61, 81, 101]
for latent_dim in [21, 41, 61, 81, 101]
    println("Benchmarking VAE latent_dim = $latent_dim...")

    model, opt_state, ps, st, st_rng, β, x_test = setup_vae_model(latent_dim)

    train_step = VAETrainStep(model, β)

    b = @benchmark begin
        result = f(
            $opt_state,
            $ps,
            $st,
            $x_test,
            $st_rng
        )
        Reactant.synchronize(result)
    end setup = (
        f = Reactant.@compile sync = true benchmark_vae_train(
            $train_step,
            $opt_state,
            $ps,
            $st,
            $x_test,
            $st_rng
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

CSV.write("benches/results/vae_latent_dim.csv", results)
println("Results saved to vae_latent_dim.csv")
println(results)
