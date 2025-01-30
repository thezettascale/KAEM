using ConfParser, Random

conf_nist = ConfParse("nist_config.ini")
parse_conf!(conf_nist)

conf_cnn = ConfParse("cnn_config.ini")
parse_conf!(conf_cnn)

ENV["GPU"] = retrieve(conf_nist, "TRAINING", "use_gpu") 
ENV["FULL_QUANT"] = retrieve(conf_nist, "MIXED_PRECISION", "full_precision")
ENV["HALF_QUANT"] = retrieve(conf_nist, "MIXED_PRECISION", "reduced_precision")

include("src/ML_pipeline/trainer.jl")
using .trainer

dataset = get(ENV, "DATASET", "MNIST")

rng = Random.seed!(1)

# Thermodynamic
conf = dataset == "MNIST" || dataset == "FMNIST" ? conf_nist : conf_cnn
t = init_trainer(rng, conf, dataset)#, img_resize=(14,14))
train!(t)  
