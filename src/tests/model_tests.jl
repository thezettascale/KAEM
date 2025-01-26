using Test, Random, LinearAlgebra, Lux, ConfParser, Zygote, ComponentArrays

ENV["GPU"] = true
ENV["QUANT"] = "FP32"

include("../T-KAM/T-KAM.jl")
include("../utils.jl")
using .T_KAM_model
using .Utils

conf = ConfParse("src/tests/test_conf.ini")
parse_conf!(conf)
out_dim = parse(Int, retrieve(conf, "KAN_LIKELIHOOD", "output_dim"))

function test_ps_derivative()
    Random.seed!(42)
    dataset = randn(quant, 3, 50) 
    model = init_T_KAM(dataset, conf)
    x_test = first(model.train_loader) |> device
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    ps, st = ComponentArray(ps) |> device, st |> device

    ∇ = first(gradient(p -> first(model.loss_fcn(model, p, st, x_test)), ps))
    @test norm(∇) > 0
    @test !any(isnan, ∇)
end

function test_grid_update()
    Random.seed!(42)
    dataset = randn(quant, 3, 50) 
    model = init_T_KAM(dataset, conf)
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    ps, st = ComponentArray(ps) |> device, st |> device

    size_grid = size(model.lkhood.Φ_fcns[Symbol("1")].grid)
    model, ps, seed = update_model_grid(model, ps, st)
    @test all(size(model.lkhood.Φ_fcns[Symbol("1")].grid) .== size_grid)
    @test !any(isnan, ps)
end

function test_mala_loss()
    Random.seed!(42)
    dataset = randn(quant, 3, 50) 
    commit!(conf, "MALA", "use_langevin", "true")
    model = init_T_KAM(dataset, conf)
    x_test = first(model.train_loader) |> device
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    ps, st = ComponentArray(ps) |> device, st |> device

    ∇ = first(gradient(p -> first(model.loss_fcn(model, p, st, x_test)), ps))
    @test norm(∇) > 0
    @test !any(isnan, ∇)
end

@testset "T-KAM Tests" begin
    test_ps_derivative()
    test_grid_update()
    test_mala_loss()
end