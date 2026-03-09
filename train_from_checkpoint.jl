"""Warning: this script will not carry over optimizer state 
or updated seed - only the current model and parameters/st_luxate."""

using JLD2, Lux, ComponentArrays, ConfParser, Random

# EDIT:
dataset = "CIFAR10"
file_loc = "logs/Vanilla/n_z=100/ULA/cnn=true/$(dataset)_1/"
ckpt = 10

conf = Dict(
    "MNIST" => ConfParse("config/nist_config.ini"),
    "FMNIST" => ConfParse("config/nist_config.ini"),
    "CIFAR10" => ConfParse("config/cifar_config.ini"),
    "SVHN" => ConfParse("config/svhn_config.ini"),
    "CELEBA" => ConfParse("config/celeba_config.ini"),
    "PTB" => ConfParse("config/text_config.ini"),
    "SMS_SPAM" => ConfParse("config/text_config.ini"),
    "DARCY_FLOW" => ConfParse("config/darcy_flow_config.ini"),
)[dataset]
parse_conf!(conf)

ENV["GPU"] = retrieve(conf, "TRAINING", "use_gpu")
ENV["PERCEPTUAL"] = retrieve(conf, "TRAINING", "use_perceptual_loss")

# EDIT:
commit!(conf, "POST_LANGEVIN", "use_langevin", "true")
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

include("src/utils.jl")
include("src/pipeline/trainer.jl")
using .Utils: pu
using .trainer

rng = Random.MersenneTwister(1)
t = init_trainer(rng, conf, dataset)

saved_data = load(file_loc * "ckpt_epoch_$ckpt.jld2")
ps_flat = saved_data["params"] .|> Float32
ps_template = Lux.initialparameters(rng, t.model)
t.ps = ComponentArray(ps_flat, getaxes(ps_template)) |> pu
t.st_kan = saved_data["kan_state"] |> pu
t.st_lux = saved_data["lux_state"] |> pu

train!(t)
