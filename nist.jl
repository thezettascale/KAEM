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

for dataset in datasets

    # Thermodynamic Integration
    commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "500")

    Random.seed!(1)
    t = init_trainer(Random.GLOBAL_RNG, conf, dataset)#, img_resize=(14,14))
    train!(t)  

    # ## Vanilla training
    # commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

    # Random.seed!(1)
    # t = init_trainer(Random.GLOBAL_RNG, conf, dataset)#, img_resize=(14,14))
    # train!(t)

end
