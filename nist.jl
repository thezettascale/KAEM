using ConfParser, Random

conf = ConfParse("nist_config.ini")
parse_conf!(conf)

ENV["GPU"] = retrieve(conf, "TRAINING", "use_gpu") 
ENV["QUANT"] = retrieve(conf, "TRAINING", "quantization")

include("src/ML_pipeline/trainer.jl")
using .trainer

datasets = [
    "MNIST", 
    "FMNIST"
    ]

num_temps = retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps")

for dataset in datasets

    commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
    commit!(conf, "MALA", "use_langevin", "false")

    # # Vanilla importance sampling
    # commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
    # Random.seed!(1)
    # t = init_trainer(Random.GLOBAL_RNG, conf, dataset)#, img_resize=(14,14))
    # train!(t)

    # MALA Vanilla
    commit!(conf, "MALA", "use_langevin", "true")
    Random.seed!(1)
    t = init_trainer(Random.GLOBAL_RNG, conf, dataset)#, img_resize=(14,14))
    train!(t)

    # Particle filter
    commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", num_temps)
    Random.seed!(1)
    t = init_trainer(Random.GLOBAL_RNG, conf, dataset)#, img_resize=(14,14))
    train!(t)  
end
