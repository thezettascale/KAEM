using ConfParser, Random

dataset = get(ENV, "DATASET", "MNIST")

conf = Dict(
    "MNIST" => ConfParse("config/nist_config.ini"),
    "FMNIST" => ConfParse("config/nist_config.ini"),
    "CIFAR10" => ConfParse("config/cifar_config.ini"),
    "SVHN" => ConfParse("config/svhn_config.ini"),
    "CIFAR10PANG" => ConfParse("config/cifar_pang_config.ini"),
    "CELEBA" => ConfParse("config/celeba_config.ini"),
    "CELEBAPANG" => ConfParse("config/celeba_pang_config.ini"),
    "SVHNPANG" => ConfParse("config/svhn_pang_config.ini"),
    "PTB" => ConfParse("config/text_config.ini"),
    "SMS_SPAM" => ConfParse("config/text_config.ini"),
    "DARCY_FLOW" => ConfParse("config/darcy_flow_config.ini"),
)[dataset]
parse_conf!(conf)

ENV["GPU"] = retrieve(conf, "TRAINING", "use_gpu")
ENV["FULL_QUANT"] = retrieve(conf, "MIXED_PRECISION", "full_precision")
ENV["HALF_QUANT"] = retrieve(conf, "MIXED_PRECISION", "reduced_precision")

include("src/pipeline/trainer.jl")
using .trainer

commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

prior_type = Dict(1 => "lognormal", 2 => "gaussian", 3 => "uniform", 4 => "ebm")
bases = Dict(5 => "RBF", 6 => "FFT")
acts = Dict(5 => "silu", 6 => "silu")
grid_sizes = Dict(5 => "20", 6 => "50")

if dataset == "CIFAR10" ||
   dataset == "SVHN" ||
   dataset == "CIFAR10PANG" ||
   dataset == "SVHNPANG" ||
   dataset == "CELEBA" ||
   dataset == "CELEBAPANG"
    rng = Random.MersenneTwister(1)
    im_resize = dataset == "CELEBA" || dataset == "CELEBAPANG" ? (64, 64) : (32, 32)
    t = init_trainer(rng, conf, dataset; img_resize = im_resize)
    train!(t)
else
    commit!(conf, "POST_LANGEVIN", "use_langevin", "false")
    for prior_idx in [3, 2, 1, 4]
        commit!(conf, "EbmModel", "π_0", prior_type[prior_idx])
        for base_idx in [5, 6]
            commit!(conf, "EbmModel", "spline_function", bases[base_idx])
            commit!(conf, "GeneratorModel", "spline_function", bases[base_idx])
            commit!(conf, "GeneratorModel", "base_activation", acts[base_idx])
            commit!(conf, "EbmModel", "base_activation", acts[base_idx])
            commit!(conf, "GeneratorModel", "grid_size", grid_sizes[base_idx])
            commit!(conf, "EbmModel", "grid_size", grid_sizes[base_idx])
            rng = Random.MersenneTwister(1)
            t = init_trainer(rng, conf, dataset)
            train!(t)
        end
    end
end
