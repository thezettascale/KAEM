using ConfParser, Random

model_type = Symbol(lowercase(get(ENV, "MODEL", "vae")))
dataset_name = get(ENV, "DATASET", "MNIST")

println("Init baseline.jl training: $model_type on $dataset_name")

conf = ConfParse("config/baseline_config.ini")
parse_conf!(conf)

include("src/baseline/baseline.jl")
using .Baseline

img_resize = nothing
if dataset_name == "CELEBA"
    img_resize = (64, 64)
end

rng = Random.MersenneTwister(42)

println("Initializing $model_type trainer for $dataset_name...")
trainer = init_baseline_trainer(
    model_type,
    conf,
    dataset_name;
    img_resize = img_resize,
    rng = rng,
)

println("Starting training...")
train!(trainer)

println("Training complete. Results saved to: $(trainer.file_loc)")
