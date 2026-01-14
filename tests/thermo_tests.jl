using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays

ENV["THERMO"] = "true"
ENV["DEVICE"] = "gpu"

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .KAEM_model

include("../src/KAEM/model_setup.jl")
using .ModelSetup

include("../src/pipeline/optimizer.jl")
using .optimization

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "4")
out_dim = parse(Int, retrieve(conf, "GeneratorModel", "output_dim"))
optimizer = create_opt(conf)
rng = Random.MersenneTwister(1)

function test_model_derivative()
    Random.seed!(42)
    dataset = randn(rng, Float32, 32, 32, 1, 500)
    model = init_KAEM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> pu
    model, opt_state, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; rng = rng)

    ps_before = Array(ps)
    loss, ps, _, st_ebm, st_gen =
        model.train_step(opt_state, ps, st_kan, st_lux, x_test, 1, st_rng)

    ps_after = Array(ps)

    @test any(ps_before .!= ps_after)
    @test !any(isnan, ps_after)

    param_change = norm(ps_after - ps_before)
    param_norm = norm(ps_before)
    relative_change = param_change / (param_norm + 1.0f-8)
    @test relative_change > 1.0f-7 # Grads flowing
    @test relative_change < 5.0f0 # Not exploding
    @test isfinite(Float32(loss))
    return @test abs(Float32(loss)) < 1.0f6
end

@testset "Thermodynamic Integration Tests" begin
    test_model_derivative()
end
