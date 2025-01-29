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

datasets = [
    "MNIST", 
    "FMNIST",
    "CIFAR10",
    "SVHN",
    ]

rng = Random.seed!(1)

# Vanilla importance sampling
for dataset in datasets
    conf = dataset == "MNIST" || dataset == "FMNIST" ? conf_nist : conf_cnn

    num_temps = retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps")

    commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
    commit!(conf, "MALA", "use_langevin", "false")

    commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
    t = init_trainer(rng, conf, dataset)#, img_resize=(14,14))
    train!(t)
end

# Thermodynamic
for datasets in datasets
    conf = dataset == "MNIST" || dataset == "FMNIST" ? conf_nist : conf_cnn
    num_temps = retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps")
    commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", num_temps)
    t = init_trainer(rng, conf, dataset)#, img_resize=(14,14))
    train!(t)  
end
