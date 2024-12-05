using ConfParser, Random

conf = ConfParse("config.ini")
parse_conf!(conf)

ENV["GPU"] = retrieve(conf, "TRAINING", "use_gpu") 

include("src/ML_pipeline/trainer.jl")
using .trainer

Random.seed!(1)
t = init_trainer(Random.GLOBAL_RNG, conf, "MNIST")
train!(t)