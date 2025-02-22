using ConfParser, Random

dataset = get(ENV, "DATASET", "MNIST")

conf = Dict(
    "MNIST" => ConfParse("config/nist_config.ini"),
    "FMNIST" => ConfParse("config/nist_config.ini"),
    "CIFAR10" => ConfParse("config/cnn_config.ini"),
    "SVHN" => ConfParse("config/cnn_config.ini"),
    "PTB" => ConfParse("config/text_config.ini"),
    "SMS_SPAM" => ConfParse("config/text_config.ini"),
    "DARCY_PERM" => ConfParse("config/darcy_perm_config.ini"),
    "DARCY_FLOW" => ConfParse("config/darcy_flow_config.ini"),
)[dataset]
parse_conf!(conf)

ENV["GPU"] = retrieve(conf, "TRAINING", "use_gpu") 
ENV["FULL_QUANT"] = retrieve(conf, "MIXED_PRECISION", "full_precision")
ENV["HALF_QUANT"] = retrieve(conf, "MIXED_PRECISION", "reduced_precision")

include("src/ML_pipeline/trainer.jl")
using .trainer


# Vanilla importance sampling
commit!(conf, "MALA", "use_langevin", "false")
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

priors = ["gaussian"]
fncs = ["RBF", "FFT"]

for prior in priors
    for fnc in fncs
        commit!(conf, "MIX_PRIOR", "π_0", prior)
        commit!(conf, "MIX_PRIOR", "spline_function", fnc)
        commit!(conf, "KAN_LIKELIHOOD", "spline_function", fnc)

        if fnc == "RBF"
            commit!(conf, "KAN_LIKELIHOOD", "base_activation", "silu")
            commit!(conf, "MIX_PRIOR", "base_activation", "silu")
        else
            commit!(conf, "KAN_LIKELIHOOD", "base_activation", "none")
            commit!(conf, "MIX_PRIOR", "base_activation", "none")
        end
        
        rng = Random.seed!(1)
        t = init_trainer(rng, conf, dataset)
        train!(t)
    end
end


