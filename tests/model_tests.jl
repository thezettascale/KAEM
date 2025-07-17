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

function test_ps_derivative()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 1, 50)
    model = init_T_KAM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> device
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    ps, st = ComponentArray(ps) |> device, st |> device
    ∇ = zero(half_quant.(ps))
    model = move_to_hq(model)

    loss_compiled = compile_mlir(model, ps, st, x_test, ∇, Random.default_rng())

    loss, ∇, st_ebm, st_gen = loss_compiled(half_quant.(ps), ∇, st, model, x_test)
    @test norm(∇) > 0
    @test !any(isnan, ∇)
end

function test_grid_update()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 1, 50)
    model = init_T_KAM(dataset, conf, (32, 32, 1))
    ps, st = Lux.setup(Random.default_rng(), model)
    ps, st = ComponentArray(ps) |> device, st |> device
    model = move_to_hq(model)

    size_grid = size(st.ebm[Symbol("1")].grid)
    x = first(model.train_loader) |> device
    model, ps, st, seed = update_model_grid(model, x, ps, Lux.testmode(st))
    @test all(size(st.ebm[Symbol("1")].grid) .== size_grid)
    @test !any(isnan, ps)
end

function test_mala_loss()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 1, 50)
    commit!(conf, "POST_LANGEVIN", "use_langevin", "true")
    model = init_T_KAM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> device
    ps, st = Lux.setup(Random.default_rng(), model)
    ps, st = ComponentArray(ps) |> device, st |> device
    model = move_to_hq(model)
    ∇ = zero(half_quant.(ps))

    loss_compiled = compile_mlir(model, ps, st, x_test, ∇, Random.default_rng())
    loss, ∇, st_ebm, st_gen = loss_compiled(half_quant.(ps), ∇, st, model, x_test)
    @test norm(∇) > 0
    @test !any(isnan, ∇)
end

function test_cnn_loss()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 3, 50)
    commit!(conf, "CNN", "use_cnn_lkhood", "true")
    model = init_T_KAM(dataset, conf, (32, 32, 3))
    x_test = first(model.train_loader) |> device
    ps, st = Lux.setup(Random.default_rng(), model)
    ps, st = ComponentArray(ps) |> device, st |> device
    model = move_to_hq(model)
    ∇ = zero(half_quant.(ps))

    loss_compiled = compile_mlir(model, ps, st, x_test, ∇, Random.default_rng())
    loss, ∇, st_ebm, st_gen = loss_compiled(half_quant.(ps), ∇, st, model, x_test)
    @test norm(∇) > 0
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
    ps, st = Lux.setup(Random.default_rng(), model)
    ps, st = ComponentArray(ps) |> device, st |> device
    model = move_to_hq(model)
    ∇ = zero(half_quant.(ps))

    loss_compiled = compile_mlir(model, ps, st, x_test, ∇, Random.default_rng())
    loss, ∇, st_ebm, st_gen = loss_compiled(half_quant.(ps), ∇, st, model, x_test)
    @test norm(∇) > 0
    @test !any(isnan, ∇)
end

@testset "T-KAM Tests" begin
    test_ps_derivative()
    test_grid_update()
    test_mala_loss()
    test_cnn_loss()
    test_seq_loss()
end
