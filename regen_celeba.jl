using ConfParser, Random, JLD2, ComponentArrays, HDF5, Lux, Reactant

dataset = "CELEBA"
file_loc = "logs/Vanilla/CELEBA/ULA/mixture/"

conf = ConfParse("config/celeba_config.ini")
parse_conf!(conf)

# Match main.jl vanilla mode settings
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

ENV["DEVICE"] = retrieve(conf, "TRAINING", "device")
ENV["PERCEPTUAL"] = retrieve(conf, "TRAINING", "use_perceptual_loss")

include("src/utils.jl")
using .Utils

include("src/pipeline/trainer.jl")
using .trainer
using .trainer: seed_rand

rng = Random.MersenneTwister(1)
t = init_trainer(rng, conf, dataset; img_resize = (64, 64), file_loc = file_loc, save_model = false)

# Load saved model - reconstruct ComponentArray using model's parameter axes
saved_data = load(file_loc * "saved_model.jld2")
ps_flat = saved_data["params"] .|> Float32
ps_template = Lux.initialparameters(rng, t.model)
t.ps = ComponentArray(ps_flat, getaxes(ps_template)) |> pu
t.st_kan = saved_data["kan_state"] |> pu
t.st_lux = saved_data["lux_state"] |> pu

# Compile generation function
println("Compiling gen...")
t.st_rng = seed_rand(t.model; rng = t.rng)
gen_compiled = Reactant.@compile t.model(
    t.ps,
    t.st_kan,
    Lux.testmode(t.st_lux),
    t.st_rng,
)
println("gen compiled.")

# Generate samples
num_batches = t.num_generated_samples ÷ t.model.batch_size
concat_dim = length(t.model.lkhood.x_shape) + 1

t.st_rng = seed_rand(t.model; rng = t.rng)
first_batch, _, _ = gen_compiled(
    t.ps,
    t.st_kan,
    Lux.testmode(t.st_lux),
    t.st_rng,
)
first_batch = Array(first_batch)

batches_to_cat = Vector{typeof(first_batch)}()
sizehint!(batches_to_cat, num_batches)
push!(batches_to_cat, first_batch)

for i in 2:num_batches
    t.st_rng = seed_rand(t.model; rng = t.rng)
    batch, _, _ = gen_compiled(
        t.ps,
        t.st_kan,
        Lux.testmode(t.st_lux),
        t.st_rng,
    )
    push!(batches_to_cat, Array(batch))
    println("Batch $i/$num_batches done")
end

gen_data = cat(batches_to_cat..., dims = concat_dim)

# Save
out_path = file_loc * "generated_images.h5"
println("Saving $(size(gen_data)) to $out_path")
try
    h5write(out_path, "samples", gen_data)
catch
    rm(out_path)
    h5write(out_path, "samples", gen_data)
end
println("Done.")
