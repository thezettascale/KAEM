"""Warning: this script will not carry over optimizer state 
or updated seed - only the current model and parameters/lux_state."""

using JLD2, Lux, LuxCUDA, CUDA, ComponentArrays, ConfParser, Random

# EDIT:
dataset = "CIFAR10"
file_loc = "logs/Vanilla/n_z=100/ULA/cnn=true/$(dataset)_1/"
ckpt = 10

conf = Dict(
    "MNIST" => ConfParse("config/nist_config.ini"),
    "FMNIST" => ConfParse("config/nist_config.ini"),
    "CIFAR10" => ConfParse("config/cifar_config.ini"),
    "SVHN" => ConfParse("config/svhn_config.ini"),
    "PTB" => ConfParse("config/text_config.ini"),
    "SMS_SPAM" => ConfParse("config/text_config.ini"),
    "DARCY_PERM" => ConfParse("config/darcy_perm_config.ini"),
    "DARCY_FLOW" => ConfParse("config/darcy_flow_config.ini"),
)[dataset]
parse_conf!(conf)

ENV["GPU"] = retrieve(conf, "TRAINING", "use_gpu")
ENV["FULL_QUANT"] = retrieve(conf, "MIXED_PRECISION", "full_precision")
ENV["HALF_QUANT"] = retrieve(conf, "MIXED_PRECISION", "reduced_precision")
ENV["AD_BACKEND"] = retrieve(conf, "TRAINING", "autodiff")

# EDIT:
commit!(conf, "POST_LANGEVIN", "use_langevin", "true")
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

include("src/utils.jl")
include("src/pipeline/trainer.jl")
using .Utils: hq, device
using .trainer

saved_data = load(file_loc * "ckpt_epoch_$ckpt.jld2")
ps = convert(ComponentArray, saved_data["params"]) |> hq |> device
st = convert(NamedTuple, saved_data["state"]) |> hq |> device

rng = Random.seed!(1)
t = init_trainer(rng, conf, dataset)
t.ps, t.st = ps, st

train!(t)
