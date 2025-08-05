using ConfParser, Random

dataset = get(ENV, "DATASET", "MNIST")

conf = Dict(
    "MNIST" => ConfParse("config/nist_config.ini"),
    "FMNIST" => ConfParse("config/nist_config.ini"),
    "CIFAR10" => ConfParse("config/cifar_config.ini"),
    "SVHN" => ConfParse("config/svhn_config.ini"),
    "PTB" => ConfParse("config/text_config.ini"),
    "SMS_SPAM" => ConfParse("config/text_config.ini"),
    "DARCY_FLOW" => ConfParse("config/darcy_flow_config.ini"),
)[dataset]
parse_conf!(conf)

ENV["THERMO"] = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps")) > 1 ? "true" : "false"
ENV["GPU"] = retrieve(conf, "TRAINING", "use_gpu")
ENV["FULL_QUANT"] = retrieve(conf, "MIXED_PRECISION", "full_precision")
ENV["HALF_QUANT"] = retrieve(conf, "MIXED_PRECISION", "reduced_precision")
ENV["autoMALA"] = retrieve(conf, "POST_LANGEVIN", "use_autoMALA")

include("src/pipeline/trainer.jl")
using .trainer

rng = Random.MersenneTwister(1)

# Thermodynamic
t = init_trainer(rng, conf, dataset)
train!(t)
