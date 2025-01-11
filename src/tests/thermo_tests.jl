using Test, Random, LinearAlgebra, Lux, ConfParser, Zygote, ComponentArrays

ENV["GPU"] = true
ENV["QUANT"] = "FP32"

include("../LV-KAM/LV-KAM.jl")
include("../utils.jl")
using .LV_KAM_model
using .Utils

conf = ConfParse("src/tests/test_conf.ini")
parse_conf!(conf)
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "30")
out_dim = parse(Int, retrieve(conf, "KAN_LIKELIHOOD", "output_dim"))
    
function test_model_derivative()
    Random.seed!(42)
    dataset = randn(quant, 3, 50) 
    model = init_LV_KAM(dataset, conf)
    x_test = first(model.train_loader) |> device
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    ps, st = ComponentArray(ps) |> device, st |> device

    ∇ = first(gradient(p -> first(MLE_loss(model, p, st, x_test)), ps))
    @test norm(∇) > 0
    @test !any(isnan, ∇)
end

@testset "Thermodynamic Integration Tests" begin
    test_model_derivative()
end