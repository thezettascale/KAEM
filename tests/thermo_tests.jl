using Test, Random, LinearAlgebra, Lux, ConfParser, Enzyme, ComponentArrays

ENV["GPU"] = false
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

function test_model_derivative()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 1, 50)
    model = init_T_KAM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> device
    ps, st = Lux.setup(Random.default_rng(), model)
    ps, st = ComponentArray(ps) |> device, st |> device
    model = prep_model(model, ps, st, x_test)
    ∇ = zero(half_quant.(ps))

    loss, ∇, st_ebm, st_gen = model.loss_fcn(half_quant.(ps), ∇, st, model, x_test)
    @test norm(∇) > 0
    @test !any(isnan, ∇)
end

@testset "Thermodynamic Integration Tests" begin
    test_model_derivative()
end
