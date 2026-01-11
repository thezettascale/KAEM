using ConfParser, Random

model_type = Symbol(lowercase(get(ENV, "MODEL", "vae")))
dataset_name = get(ENV, "DATASET", "MNIST")

println("Init baseline training: $model_type on $dataset_name")

conf = Dict(
    "MNIST" => ConfParse("config/baseline_nist_config.ini"),
    "FMNIST" => ConfParse("config/baseline_nist_config.ini"),
    "CIFAR10" => ConfParse("config/baseline_cifar_config.ini"),
    "SVHN" => ConfParse("config/baseline_svhn_config.ini"),
    "CELEBA" => ConfParse("config/baseline_celeba_config.ini"),
)[dataset_name]
parse_conf!(conf)

include("src/baseline/training/trainer.jl")
using .Baseline: init_trainer, train!

img_resize = Dict(
    "MNIST" => nothing,
    "FMNIST" => nothing,
    "CIFAR10" => (32, 32),
    "SVHN" => (32, 32),
    "CELEBA" => (64, 64),
)[dataset_name]

rng = Random.MersenneTwister(42)

println("Initializing $model_type trainer for $dataset_name...")
trainer = init_trainer(
    model_type,
    conf,
    dataset_name;
    img_resize = img_resize,
    rng = rng,
)

println("Starting training...")
train!(trainer)

println("Training complete. Results saved to: $(trainer.file_loc)")
