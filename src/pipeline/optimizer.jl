module optimization

export opt, create_opt

using Lux, ConfParser, Optimisers, ParameterSchedulers

include("manual_adam.jl")
using .AdamOptimizer

struct opt
    rule
    schedule
end

function create_opt(conf::ConfParse)
    """
    Create an optimizer from a configuration file.

    Args:
        conf: ConfParse object

    Returns:
        opt: opt object, which initializes the optimizer when called
    """

    LR = parse(Float32, retrieve(conf, "OPTIMIZER", "learning_rate"))
    β = parse.(Float32, retrieve(conf, "OPTIMIZER", "betas")) |> Tuple
    decay = parse(Float32, retrieve(conf, "OPTIMIZER", "decay"))
    ρ = parse(Float32, retrieve(conf, "OPTIMIZER", "ρ"))
    ε = parse(Float32, retrieve(conf, "OPTIMIZER", "ε"))

    opt_type = retrieve(conf, "OPTIMIZER", "type")

    opt_mapping = Dict(
        "adam" => () -> ManualAdam(LR, β, 0.0f0, ε),
        "adamw" => () -> ManualAdam(LR, β, decay, ε),
        "momentum" => () -> Optimisers.Momentum(LR, ρ),
        "nesterov" => () -> Optimisers.Nesterov(LR, ρ),
        "sgd" => () -> Optimisers.Descent(LR),
        "rmsprop" => () -> Optimisers.RMSProp(LR, ρ, ε)
    )

    optimizer = get(opt_mapping, opt_type, () -> Optimisers.Adam(LR, β, ε))

    gamma = parse(Float32, retrieve(conf, "LR_SCHEDULE", "decay"))
    milestones = parse.(Float32, retrieve(conf, "LR_SCHEDULE", "milestone_epochs"))

    b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
    dataset_size = parse(Int, retrieve(conf, "TRAINING", "N_train"))
    num_params_updates = ceil(dataset_size / b_size)
    milestones .*= num_params_updates

    schedule = ParameterSchedulers.Step(LR, gamma, milestones)
    return opt(optimizer, ParameterSchedulers.Stateful(schedule))
end

end
