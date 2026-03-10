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

# Load saved params/states FIRST
saved_data = load(file_loc * "saved_model.jld2")
ps_flat = saved_data["params"] .|> Float32
st_kan_saved = saved_data["kan_state"]
st_lux_saved = saved_data["lux_state"]

# Init trainer — this creates a fresh model with fresh params
rng = Random.MersenneTwister(1)
t = init_trainer(rng, conf, dataset; img_resize = (64, 64), file_loc = file_loc, save_model = false)

# Reconstruct ComponentArray with correct axes from the model, then swap in saved values
t.ps = ComponentArray(ps_flat, getaxes(t.ps)) |> pu
t.st_kan = st_kan_saved |> pu
t.st_lux = st_lux_saved |> pu

# Compile gen AFTER loading saved params/states
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
batch_size = t.model.batch_size
concat_dim = length(t.model.lkhood.x_shape) + 1
total_shape = (t.model.lkhood.x_shape..., num_batches * batch_size)

# Write directly to h5 batch-by-batch to avoid memory pressure
gen_path = file_loc * "generated_images.h5"
isfile(gen_path) && rm(gen_path)

h5open(gen_path, "w") do fid
    dset = create_dataset(fid, "samples", Float32, total_shape)
    for i in 1:num_batches
        t.st_rng = seed_rand(t.model; rng = t.rng)
        batch, _, _ = gen_compiled(
            t.ps,
            t.st_kan,
            Lux.testmode(t.st_lux),
            t.st_rng,
        )
        idx_start = (i - 1) * batch_size + 1
        idx_end = i * batch_size
        selectdim(dset, concat_dim, idx_start:idx_end) .= Array(batch)
        println("Batch $i/$num_batches done")
    end
end
println("Done. Saved to $gen_path")
