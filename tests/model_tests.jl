using Test, Random, LinearAlgebra, Lux, ConfParser, Enzyme, ComponentArrays

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/T-KAM/T-KAM.jl")
include("../src/T-KAM/kan/grid_updating.jl")
include("../src/utils.jl")
using .T_KAM_model
using .GridUpdating: update_model_grid
using .Utils

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
out_dim = parse(Int, retrieve(conf, "GeneratorModel", "output_dim"))
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

function test_ps_derivative()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 1, 50)
    model = init_T_KAM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> device
    model, ps, st_kan, st_lux = prep_model(model, x_test)
    ps = half_quant.(ps)
    ∇ = Enzyme.make_zero(ps)

    loss, ∇, st_ebm, st_gen =
        model.loss_fcn(ps, ∇, st_kan, st_lux, model, x_test; rng = Random.default_rng())

    @test norm(∇) != 0
    @test !any(isnan, ∇)
end

function test_grid_update()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 1, 50)
    model = init_T_KAM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> device
    model, ps, st_kan, st_lux = prep_model(model, x_test)
    ps = half_quant.(ps)

    size_grid = size(st_kan.ebm[:a].grid)
    x = first(model.train_loader) |> device
    model, ps, st_kan, st_lux =
        update_model_grid(model, x, ps, st_kan, Lux.testmode(st_lux))
    @test all(size(st_kan.ebm[:a].grid) .== size_grid)
    @test !any(isnan, ps)
end

function test_mala_loss()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 1, 50)
    commit!(conf, "POST_LANGEVIN", "use_langevin", "true")
    model = init_T_KAM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> device
    model, ps, st_kan, st_lux = prep_model(model, x_test)
    ps = half_quant.(ps)
    ∇ = Enzyme.make_zero(ps)

    loss, ∇, st_ebm, st_gen =
        model.loss_fcn(ps, ∇, st_kan, st_lux, model, x_test; rng = Random.default_rng())
    @test norm(∇) != 0
    @test !any(isnan, ∇)
end

function test_cnn_loss()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 3, 50)
    commit!(conf, "CNN", "use_cnn_lkhood", "true")
    model = init_T_KAM(dataset, conf, (32, 32, 3))
    x_test = first(model.train_loader) |> device
    model, ps, st_kan, st_lux = prep_model(model, x_test)
    ps = half_quant.(ps)
    ∇ = Enzyme.make_zero(ps)

    loss, ∇, st_ebm, st_gen =
        model.loss_fcn(ps, ∇, st_kan, st_lux, model, x_test; rng = Random.default_rng())
    @test norm(∇) != 0
    @test !any(isnan, ∇)
    commit!(conf, "CNN", "use_cnn_lkhood", "false")
end

function test_seq_loss()
    Random.seed!(42)
    dataset = randn(full_quant, 50, 10, 100)
    commit!(conf, "SEQ", "sequence_length", "10")
    commit!(conf, "SEQ", "vocab_size", "50")
    model = init_T_KAM(dataset, conf, (50, 10))
    x_test = first(model.train_loader) |> device
    model, ps, st_kan, st_lux = prep_model(model, x_test)
    ps = half_quant.(ps)
    ∇ = Enzyme.make_zero(ps)

    loss, ∇, st_ebm, st_gen =
        model.loss_fcn(ps, ∇, st_kan, st_lux, model, x_test; rng = Random.default_rng())
    @test norm(∇) != 0
    @test !any(isnan, ∇)
end

@testset "T-KAM Tests" begin
    test_ps_derivative()
    test_grid_update()
    test_mala_loss()
    test_cnn_loss()
    test_seq_loss()
end
