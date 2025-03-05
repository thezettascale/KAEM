using BenchmarkTools, ConfParser, Lux, Zygote, Random, CUDA

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP16"

include("../src/T-KAM/T-KAM.jl")
using .T_KAM_model

conf = ConfParse("img_config.ini")
parse_conf!(conf)

function benchmark_dim(n_z)
    commit!(conf, "EBM_PRIOR", "layer_widths", "$(n_z), $(2*n_z+1)")
    commit!(conf, "KAN_LIKELIHOOD", "layer_widths", "$(2*n_z+1)")

    Random.seed!(42)
    dataset = randn(Float32, 784, 10000) 
    model = init_T_KAM(dataset, conf)
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    x_test = first(model.train_loader) 

    first(gradient(p -> first(model.loss_fcn(model, p, st, x_test)), ps))
end

@benchmark CUDA.@sync benchmark_dim(10)
@benchmark CUDA.@sync benchmark_dim(50)
@benchmark CUDA.@sync benchmark_dim(100)