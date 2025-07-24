using Test, Random, LinearAlgebra, Lux, ConfParser, Enzyme, ComponentArrays

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/T-KAM/T-KAM.jl")
include("../src/T-KAM/model_setup.jl")
include("../src/T-KAM/loss_fcns/thermodynamic.jl")
include("../src/utils.jl")
using .T_KAM_model
using .ThermodynamicIntegration: sample_thermo
using .ModelSetup: prep_model
using .Utils

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "4")
out_dim = parse(Int, retrieve(conf, "GeneratorModel", "output_dim"))

function test_posterior_sampling()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 1, 50)
    model = init_T_KAM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> device
    model, ps, st_kan, st_lux = prep_model(model, x_test)
    ps = half_quant.(ps)

    z_posterior, temps, st_lux = sample_thermo(
        ps,
        st_kan,
        st_lux,
        model,
        x_test;
        train_idx = 1,
        rng = Random.default_rng(),
    )
    @test size(z_posterior) == (10, 5, 10, 4)
    @test size(temps) == (4,)
    @test !any(isnan, z_posterior)
end

function test_model_derivative()
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

@testset "Thermodynamic Integration Tests" begin
    test_posterior_sampling()
    test_model_derivative()
end
