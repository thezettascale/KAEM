using ConfParser, Random

dataset = get(ENV, "DATASET", "MNIST")
conf = (dataset == "MNIST" || dataset == "FMNIST") ? ConfParse("nist_config.ini") : ConfParse("cnn_config.ini")
conf = (dataset == "PTB" || dataset == "SMS_SPAM") ? ConfParse("text_config.ini") : conf
parse_conf!(conf)

ENV["GPU"] = retrieve(conf, "TRAINING", "use_gpu") 
ENV["FULL_QUANT"] = retrieve(conf, "MIXED_PRECISION", "full_precision")
ENV["HALF_QUANT"] = retrieve(conf, "MIXED_PRECISION", "reduced_precision")

include("src/ML_pipeline/trainer.jl")
using .trainer

rng = Random.seed!(1)

# Vanilla importance sampling
commit!(conf, "MALA", "use_langevin", "false")
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

t = init_trainer(rng, conf, dataset)
train!(t)

