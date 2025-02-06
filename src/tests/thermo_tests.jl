using Test, Random, LinearAlgebra, Lux, ConfParser, Zygote, ComponentArrays

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP16"

include("../T-KAM/T-KAM.jl")
include("../utils.jl")
using .T_KAM_model
using .Utils

conf = ConfParse("src/tests/test_conf.ini")
parse_conf!(conf)
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "10")
out_dim = parse(Int, retrieve(conf, "KAN_LIKELIHOOD", "output_dim"))
    
function test_loss()
    Random.seed!(42)
    dataset = randn(full_quant, 3, 3, 1, 50) 
    model = init_T_KAM(dataset, conf, (3,3,1))
    x_test = first(model.train_loader) |> device
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    ps, st = ComponentArray(ps) |> device, st |> device

    loss = first(model.loss_fcn(model, half_quant.(ps), st, x_test))
    @test loss != 0
    @test !isnan(loss)
end

function test_model_derivative()
    Random.seed!(42)
    dataset = randn(full_quant, 3, 3, 1, 50) 
    model = init_T_KAM(dataset, conf, (3,3,1))
    x_test = first(model.train_loader) |> device
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    ps, st = ComponentArray(ps) |> device, st |> device

    ∇ = first(gradient(p -> first(model.loss_fcn(model, p, st, x_test)), half_quant.(ps)))
    @test norm(∇) > 0
    @test !any(isnan, ∇)
end

@testset "Thermodynamic Integration Tests" begin
    test_loss()
    test_model_derivative()
end