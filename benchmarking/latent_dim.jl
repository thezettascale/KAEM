using BenchmarkTools, ConfParser, Lux, Zygote, Random

include("../src/LV-KAM/LV-KAM.jl")
using .LV_KAM_model

conf = ConfParse("nist_config.ini")
parse_conf!(conf)

function benchmark_dim(n_z)
    commit!(conf, "MIX_PRIOR", "layer_widths", "$(n_z), $(2*n_z+1)")
    commit!(conf, "MOE_LIKELIHOOD", "layer_widths", "$(2*n_z+1)")

    Random.seed!(42)
    dataset = randn(Float32, 784, 10000) 
    model = init_LV_KAM(dataset, conf)
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    x_test = first(model.train_loader) 

    first(gradient(p -> first(MLE_loss(model, p, st, x_test)), ps))
end

@benchmark benchmark_dim(10)
@benchmark benchmark_dim(50)
@benchmark benchmark_dim(100)