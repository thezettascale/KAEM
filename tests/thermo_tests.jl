using Test, Random, LinearAlgebra, Lux, ConfParser, Enzyme, ComponentArrays

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"
ENV["autoMALA"] = "true"

include("../src/T-KAM/T-KAM.jl")
include("../src/utils.jl")
using .T_KAM_model
using .Utils

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "4")
out_dim = parse(Int, retrieve(conf, "GeneratorModel", "output_dim"))

function test_loss()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 1, 50)
    model = init_T_KAM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> device
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    ps, st = ComponentArray(ps) |> device, st |> device

    loss_compiled = compile_mlir(model, ps, st, x_test, ∇)
    loss = first(loss_compiled(half_quant.(ps), st, model, x_test))
    @test !isnan(loss)
end

function test_model_derivative()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 1, 50)
    model = init_T_KAM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> device
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    ps, st = ComponentArray(ps) |> device, st |> device
    ∇ = zero(half_quant.(ps))

    loss_compiled = compile_mlir(model, ps, st, x_test, ∇)
    loss, ∇, st_ebm, st_gen, seed =
        loss_compiled(half_quant.(ps), ∇, st, model, x_test; seed = 1)
    @test norm(∇) > 0
    @test !any(isnan, ∇)
end

@testset "Thermodynamic Integration Tests" begin
    test_loss()
    test_model_derivative()
end
