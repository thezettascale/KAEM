"""Warning: this script will not carry over optimizer state 
or updated seed, only the current, model, and parameters"""

using JLD2, Lux, LuxCUDA, CUDA, ComponentArrays, ConfParser

# EDIT:
file_loc = "logs/Thermodynamic/CIFAR10_1/"
ckpt = 10

conf = (occursin("MNIST", x)|| occursin("FMNIST", x)) ? ConfParse("nist_config.ini") : ConfParse("cnn_config.ini")
conf = (occursin("PTB", x) || occursin("SMS_SPAM", x)) ? ConfParse("text_config.ini") : conf

ENV["GPU"] = retrieve(conf, "TRAINING", "use_gpu") 
ENV["FULL_QUANT"] = retrieve(conf, "MIXED_PRECISION", "full_precision")
ENV["HALF_QUANT"] = retrieve(conf, "MIXED_PRECISION", "reduced_precision")

# EDIT:
commit!(conf, "MALA", "use_langevin", "false")
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

include("src/utils.jl")
include("src/ML_pipeline/trainer.jl")
using .Utils
using .trainer


saved_data = load(file_loc * "ckpt_epoch_$ckpt.jld2")
ps = convert(ComponentArray, saved_data["params"]) |> hq |> device
st = convert(NamedTuple, saved_data["state"]) |> hq |> device

rng = Random.seed!(1)
t = init_trainer(rng, conf, "CIFAR10_1")
t.ps, t.st = ps, st

train!(t)


