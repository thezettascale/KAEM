using BenchmarkTools, ConfParser, Lux, Zygote, Random

include("../src/LV-KAM/LV-KAM.jl")
using .LV_KAM_model

conf = ConfParse("benchmarking/benchmark_conf.ini")
parse_conf!(conf)

function benchmark_temps(N_t)
    commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "$(N_t)")

    Random.seed!(42)
    dataset = randn(Float32, 784, 10000) 
    model = init_LV_KAM(dataset, conf)
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    x_test = first(model.train_loader) 

    first(gradient(p -> first(MLE_loss(model, p, st, x_test)), ps))
end

@benchmark benchmark_temps(arg) setup=(arg=10)
@benchmark benchmark_temps(arg) setup=(arg=50)
@benchmark benchmark_temps(arg) setup=(arg=100)
