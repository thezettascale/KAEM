using BenchmarkTools, ConfParser, Lux, Zygote, Random

ENV["GPU"] = true
ENV["QUANT"] = "FP32"

include("../src/T-KAM/T-KAM.jl")
using .T_KAM_model

conf = ConfParse("nist_config.ini")#
parse_conf!(conf)

function benchmark_temps(N_t)
    commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "$(N_t)")

    Random.seed!(42)
    dataset = randn(Float32, 784, 10000) 
    model = init_T_KAM(dataset, conf)
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    x_test = first(model.train_loader) 

    first(gradient(p -> first(MLE_loss(model, p, st, x_test)), ps))
end

@benchmark CUDA.@sync benchmark_temps(10) 
@benchmark CUDA.@sync benchmark_temps(50) 
@benchmark CUDA.@sync benchmark_temps(100) 
