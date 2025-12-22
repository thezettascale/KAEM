using ConfParser, Random

dataset = get(ENV, "DATASET", "MNIST")
mode = get(ENV, "MODE", "vanilla")
use_thermo = mode == "thermo"

println("Init main.jl training as $dataset $mode")

println("=== XLA Environment Config ===")
println("XLA_REACTANT_GPU_MEM_FRACTION: ", get(ENV, "XLA_REACTANT_GPU_MEM_FRACTION", "not set"))
println("XLA_REACTANT_GPU_PREALLOCATE: ", get(ENV, "XLA_REACTANT_GPU_PREALLOCATE", "not set"))
xla_flags = get(ENV, "XLA_FLAGS", "")
println("XLA_FLAGS: ", isempty(xla_flags) ? "not set" : xla_flags)
println("===================================")
println()

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

N_t = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps"))
ENV["THERMO"] = (N_t > 1 && use_thermo) ? "true" : "false"
ENV["GPU"] = retrieve(conf, "TRAINING", "use_gpu")
ENV["PERCEPTUAL"] = retrieve(conf, "TRAINING", "use_perceptual_loss")

include("src/pipeline/trainer.jl")
using .trainer

if !use_thermo || N_t <= 1
    commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
end

default_act = retrieve(conf, "EbmModel", "base_activation")

prior_type = Dict(1 => "lognormal", 2 => "gaussian", 3 => "uniform", 4 => "ebm")
bases = Dict(5 => "RBF", 6 => "Cheby", 7 => "FFT")
acts = Dict(5 => default_act, 6 => "none", 7 => default_act)
grid_sizes = Dict(5 => "20", 6 => "1", 7 => "50")

rng = Random.MersenneTwister(1)
im_resize = dataset == "CELEBA" || dataset == "CELEBAPANG" ? (64, 64) : (32, 32)

if use_thermo && N_t > 1
    t = init_trainer(rng, conf, dataset; img_resize = im_resize)
    train!(t)
else
    if dataset == "CIFAR10" ||
            dataset == "SVHN" ||
            dataset == "CIFAR10PANG" ||
            dataset == "SVHNPANG" ||
            dataset == "CELEBA" ||
            dataset == "CELEBAPANG"
        t = init_trainer(rng, conf, dataset; img_resize = im_resize)
        train!(t)
    else
        commit!(conf, "POST_LANGEVIN", "use_langevin", "false")
        for prior_idx in [4, 2, 3, 1]
            commit!(conf, "EbmModel", "π_0", prior_type[prior_idx])
            for base_idx in [5]
                commit!(conf, "EbmModel", "spline_function", bases[base_idx])
                commit!(conf, "GeneratorModel", "spline_function", bases[base_idx])
                commit!(conf, "GeneratorModel", "base_activation", acts[base_idx])
                commit!(conf, "EbmModel", "base_activation", acts[base_idx])
                commit!(conf, "GeneratorModel", "grid_size", grid_sizes[base_idx])
                commit!(conf, "EbmModel", "grid_size", grid_sizes[base_idx])
                if base_idx == 6
                    commit!(conf, "EbmModel", "τ_trainable", "false")
                    commit!(conf, "EbmModel", "init_τ", "1.001")
                    commit!(conf, "GeneratorModel", "τ_trainable", "false")
                    commit!(conf, "GeneratorModel", "init_τ", "1.001")
                end
                t = init_trainer(rng, conf, dataset)
                train!(t)

                commit!(conf, "EbmModel", "τ_trainable", "true")
                commit!(conf, "EbmModel", "init_τ", "1")
                commit!(conf, "GeneratorModel", "τ_trainable", "true")
                commit!(conf, "GeneratorModel", "init_τ", "1")
            end
        end
    end
end
