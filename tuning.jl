using ConfParser, Random, HyperTuning

dataset = get(ENV, "DATASET", "MNIST")

println("=== XLA Environment Config ===")
println("XLA_REACTANT_GPU_MEM_FRACTION: ", get(ENV, "XLA_REACTANT_GPU_MEM_FRACTION", "not set"))
println("XLA_REACTANT_GPU_PREALLOCATE: ", get(ENV, "XLA_REACTANT_GPU_PREALLOCATE", "not set"))
xla_flags = get(ENV, "XLA_FLAGS", "")
println("XLA_FLAGS: ", isempty(xla_flags) ? "not set" : xla_flags)
println("===================================")
println()

conf = Dict(
    "MNIST" => ConfParse("config/nist_tuning_config.ini"),
    "SVHN" => ConfParse("config/svhn_tuning_config.ini"),
    "CELEBA" => ConfParse("config/celeba_tuning_config.ini"),
)[dataset]
parse_conf!(conf)

ENV["THERMO"] = "false"
ENV["GPU"] = retrieve(conf, "TRAINING", "use_gpu")
ENV["PERCEPTUAL"] = retrieve(conf, "TRAINING", "use_perceptual_loss")

num_trials = parse(Int, retrieve(conf, "TUNING", "num_trials"))
sampler_type = retrieve(conf, "TUNING", "sampler")

include("src/pipeline/trainer.jl")
using .trainer

commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

rng = Random.MersenneTwister(1)
im_resize = dataset == "CELEBA" ? (64, 64) : nothing
im_resize = dataset == "SVHN" ? (32, 32) : im_resize

function objective(trial)
    @unpack (
        opt_type,
        learning_rate,
        decay,
        opt_decay,
        prior_type,
        langevin_step,
        generator_var,
        noise_var,
        basis_act,
        cnn_act,
    ) = trial

    commit!(conf, "OPTIMIZER", "type", opt_type)
    commit!(conf, "OPTIMIZER", "learning_rate", string(learning_rate))
    commit!(conf, "LR_SCHEDULE", "decay", string(decay))
    commit!(conf, "OPTIMIZER", "decay", string(opt_decay))
    commit!(conf, "EbmModel", "π_0", prior_type)
    commit!(conf, "POST_LANGEVIN", "initial_step_size", string(langevin_step))
    commit!(conf, "GeneratorModel", "generator_variance", string(generator_var))
    commit!(conf, "GeneratorModel", "generator_noise", string(noise_var))
    commit!(conf, "EbmModel", "base_activation", basis_act)
    commit!(conf, "GeneratorModel", "base_activation ", basis_act)
    commit!(conf, "CNN", "activation", cnn_act)

    t = init_trainer(rng, conf, dataset; img_tuning = true, img_resize = im_resize)
    return train!(t; trial = trial)
end

const sampler = Dict(
    "bcap" => BCAPSampler,
    "grid" => GridSampler,
    "random" => RandomSampler,
)[sampler_type]

space = Scenario(
    opt_type = ["adam", "nesterov", "adamw"],
    learning_rate = (1.0e-5 .. 1.0e-2),
    decay = 0.0e0 .. 1.0e0,
    opt_decay = 0.0e0 .. 1.0e0,
    prior_type = ["ebm", "gaussian"],
    langevin_step = 1.0e-3 .. 1.0e-1,
    generator_var = 1.0e-2 .. 1.0e0,
    noise_var = 1.0e-2 .. 1.0e0,
    basis_act = [
        "relu",
        "leakyrelu",
        "swish",
        "sigmoid",
        "gelu",
        "selu",
        "tanh",
    ],
    cnn_act = [
        "relu",
        "leakyrelu",
        "swish",
        "sigmoid",
        "gelu",
        "selu",
        "tanh",
    ],
    max_trials = num_trials,
    # pruner = MedianPruner(),
    sampler = sampler()
)

HyperTuning.optimize(objective, space)

display(top_parameters(space))
@unpack (
    opt_type,
    learning_rate,
    decay,
    opt_decay,
    prior_type,
    langevin_step,
    generator_var,
    noise_var,
    basis_act,
    cnn_act,
) = space

commit!(conf, "OPTIMIZER", "type", opt_type)
commit!(conf, "OPTIMIZER", "learning_rate", string(learning_rate))
commit!(conf, "LR_SCHEDULE", "decay", string(decay))
commit!(conf, "OPTIMIZER", "decay", string(opt_decay))
commit!(conf, "EbmModel", "π_0", prior_type)
commit!(conf, "POST_LANGEVIN", "initial_step_size", string(langevin_step))
commit!(conf, "GeneratorModel", "generator_variance", string(generator_var))
commit!(conf, "GeneratorModel", "generator_noise", string(noise_var))
commit!(conf, "EbmModel", "base_activation", basis_act)
commit!(conf, "GeneratorModel", "base_activation", basis_act)
commit!(conf, "CNN", "activation ", cnn_act)

if dataset == "MNIST"
    save!(conf, "config/nist_tuning_config.ini")
    conf = ConfParse("config/nist_config.ini")
    parse_conf!(conf)
elseif dataset == "SVHN"
    save!(conf, "config/svhn_tuning_config.ini")
    conf = ConfParse("config/svhn_config.ini")
    parse_conf!(conf)
else
    save!(conf, "config/celeba_tuning_config.ini")
    conf = ConfParse("config/celeba_config.ini")
    parse_conf!(conf)
end

commit!(conf, "OPTIMIZER", "type", opt_type)
commit!(conf, "OPTIMIZER", "learning_rate", string(learning_rate))
commit!(conf, "LR_SCHEDULE", "decay", string(decay))
commit!(conf, "OPTIMIZER", "decay", string(opt_decay))
commit!(conf, "EbmModel", "π_0", prior_type)
commit!(conf, "POST_LANGEVIN", "initial_step_size", string(langevin_step))
commit!(conf, "GeneratorModel", "generator_variance", string(generator_var))
commit!(conf, "GeneratorModel", "generator_noise", string(noise_var))
commit!(conf, "EbmModel", "base_activation", basis_act)
commit!(conf, "GeneratorModel", "base_activation", basis_act)
commit!(conf, "CNN", "activation ", cnn_act)

if dataset == "MNIST"
    save!(conf, "config/nist_config.ini")
elseif dataset == "SVHN"
    save!(conf, "config/svhn_config.ini")
else
    save!(conf, "config/celeba_config.ini")
end
