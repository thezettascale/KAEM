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

function test_encoder_init()
    x_shape = (32, 32, 1)
    encoder = init_encoder(conf, x_shape; rng = rng)

    @test encoder isa EncoderWrapper
    @test encoder.latent_dim == 5 * 10
    @test encoder.bool_config.variational == true

    ps = Lux.initialparameters(rng, encoder)
    st = Lux.initialstates(rng, encoder)

    @test haskey(ps, :layers)
    @test haskey(ps, :mu)
    @test haskey(ps, :logvar)

    return @test true
end

function test_variational_model_init()
    dataset = randn(rng, Float32, 32, 32, 1, 500)
    model = init_KAEM(dataset, conf, (32, 32, 1))

    @test model.variational == true
    @test model.encoder.bool_config.variational == true
    @test model.encoder isa EncoderWrapper

    return @test true
end

function test_variational_params_states()
    dataset = randn(rng, Float32, 32, 32, 1, 500)
    model = init_KAEM(dataset, conf, (32, 32, 1))

    ps = Lux.initialparameters(rng, model)
    st_kan, st_lux = Lux.initialstates(rng, model)

    @test haskey(ps, :ebm)
    @test haskey(ps, :gen)
    @test haskey(ps, :enc)

    @test haskey(st_lux, :ebm)
    @test haskey(st_lux, :gen)
    @test haskey(st_lux, :enc)

    return @test true
end

function test_variational_loss()
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
    @testset "Encoder Tests" begin
        test_encoder_init()
    end

    @testset "Model Integration Tests" begin
        test_variational_model_init()
        test_variational_params_states()
    end

    @testset "Training Tests" begin
        test_variational_loss()
    end
end
