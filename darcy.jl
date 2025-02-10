using ConfParser, Random

conf = ConfParse("darcy_config.ini")
parse_conf!(conf)

ENV["GPU"] = retrieve(conf, "TRAINING", "use_gpu") 
ENV["FULL_QUANT"] = retrieve(conf, "MIXED_PRECISION", "full_precision")
ENV["HALF_QUANT"] = retrieve(conf, "MIXED_PRECISION", "reduced_precision")

include("src/ML_pipeline/darcy_trainer.jl")
using .trainer

rng = Random.seed!(1)

commit!(conf, "MALA", "use_langevin", "false")
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

t = init_trainer(rng, conf, true)
train!(t)

t = init_trainer(rng, conf, false)
train!(t)
