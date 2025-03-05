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

commit!(conf, "MALA", "use_langevin", "false")
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

prior_type = Dict(
    1 => "lognormal",
    2 => "gaussian",
    3 => "uniform",
)

bases = Dict(
    4 => "RBF",
    5 => "FFT",
)

acts = Dict(
    4 => "silu",
    5 => "none",
)

if dataset == "CIFA10" || dataset == "SVHN" 
    rng = Random.seed!(1)
    t = init_trainer(rng, conf, dataset)
    train!(t)
else
    for prior_idx in [1, 2, 3]
        commit!(conf, "EBM_PRIOR", "π_0", prior_type[prior_idx])
        for base_idx in [4, 5]
            commit!(conf, "EBM_PRIOR", "spline_function", bases[base_idx])
            commit!(conf, "KAN_LIKELIHOOD", "spline_function", bases[base_idx])
            commit!(conf, "KAN_LIKELIHOOD", "base_activation", acts[base_idx])
            commit!(conf, "EBM_PRIOR", "base_activation", acts[base_idx])
            rng = Random.seed!(1)
            t = init_trainer(rng, conf, dataset)
            train!(t)
        end
    end     
end





