using Test, Random, LinearAlgebra, Lux, ConfParser, Zygote, ComponentArrays

ENV["GPU"] = true

include("../LV-KAM/LV-KAM.jl")
include("../utils.jl")
using .LV_KAM_model
using .Utils

conf = ConfParse("src/unit_tests/test_conf.ini")
parse_conf!(conf)
out_dim = parse(Int, retrieve(conf, "MOE_LIKELIHOOD", "output_dim"))

function test_ps_derivative()
    Random.seed!(42)
    dataset = randn(Float32, 3, 50) 
    model = init_LV_KAM(dataset, conf)
    x_test = first(model.train_loader) |> device
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    ps, st = ComponentArray(ps) |> device, st |> device

    ∇ = first(gradient(p -> MLE_loss(model, p, st, x_test), ps))
    @test norm(∇) > 0
    @test !any(isnan, ∇)
end

function test_grid_update()
    Random.seed!(42)
    dataset = randn(Float32, 3, 50) 
    model = init_LV_KAM(dataset, conf)
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    ps, st = ComponentArray(ps) |> device, st |> device

    size_grid = size(model.lkhood.Λ_fcns[Symbol("1")].grid)
    model, ps, seed = update_llhood_grid(model, ps, st)
    @test all(size(model.lkhood.Λ_fcns[Symbol("1")].grid) .== size_grid)
    @test !any(isnan, ps)

end

@testset "LV-KAM Tests" begin
    test_ps_derivative()
    test_grid_update()
end