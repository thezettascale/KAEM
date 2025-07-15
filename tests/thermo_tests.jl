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

    loss = first(model.loss_fcn(half_quant.(ps), st, model, x_test))
    @test !isnan(loss)
end

function test_model_derivative()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 1, 50)
    model = init_T_KAM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> device
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    ps, st = ComponentArray(ps) |> device, st |> device
    ∇ = zero(ps)

    f = (p, s, m, x) -> first(model.loss_fcn(p, s, m, x))
    Enzyme.autodiff(
        set_runtime_activity(Reverse),
        f,
        Enzyme.Active,
        Enzyme.Duplicated(ps, ∇),
        Enzyme.Const(st),
        Enzyme.Const(model),
        Enzyme.Const(x_test),
    )
    @test norm(∇) > 0
    @test !any(isnan, ∇)
end

@testset "Thermodynamic Integration Tests" begin
    test_loss()
    test_model_derivative()
end
