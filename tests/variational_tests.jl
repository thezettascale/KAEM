using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays, Reactant
using MLDataDevices: cpu_device

ENV["GPU"] = "false"
ENV["REACTANT_PREFER_CPU"] = "true"

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .KAEM_model
using .KAEM_model.EncoderModel

include("../src/KAEM/model_setup.jl")
using .ModelSetup

include("../src/pipeline/optimizer.jl")
using .optimization

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)

commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
commit!(conf, "POST_LANGEVIN", "use_langevin", "false")
commit!(conf, "VARIATIONAL", "use_variational", "true")

optimizer = create_opt(conf)
rng = Random.MersenneTwister(42)

function test_diagonal_loss()
    commit!(conf, "Encoder", "type", "diagonal")
    dataset = randn(rng, Float32, 32, 32, 1, 500)
    model = init_KAEM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> pu
    model, opt_state, ps, st_kan, st_lux, st_rng = prep_model(
        model,
        x_test,
        optimizer;
        rng = rng,
    )

    @test model.variational == true
    @test model.train_step !== nothing
    @test haskey(st_rng, :encoder_noise)

    ps_before = Array(ps)
    loss, ps, _, st_ebm, st_gen =
        model.train_step(opt_state, ps, st_kan, st_lux, x_test, 1, st_rng)

    ps_after = Array(ps)
    @test any(ps_before .!= ps_after)
    return @test !any(isnan, ps_after)
end

function test_cnn_loss()
    commit!(conf, "Encoder", "type", "cnn")
    dataset = randn(rng, Float32, 32, 32, 1, 500)
    model = init_KAEM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> pu
    model, opt_state, ps, st_kan, st_lux, st_rng = prep_model(
        model,
        x_test,
        optimizer;
        rng = rng,
    )

    @test model.variational == true
    @test model.train_step !== nothing
    @test haskey(st_rng, :encoder_noise)

    ps_before = Array(ps)
    loss, ps, _, st_ebm, st_gen =
        model.train_step(opt_state, ps, st_kan, st_lux, x_test, 1, st_rng)

    ps_after = Array(ps)
    @test any(ps_before .!= ps_after)
    return @test !any(isnan, ps_after)
end

@testset "Variational Training Tests" begin
    test_diagonal_variational_loss()
    test_cnn_variational_loss()
end
