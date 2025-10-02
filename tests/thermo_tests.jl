using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays

ENV["THERMO"] = "true"
ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP16"

include("../src/utils.jl")
using .Utils

include("../src/T-KAM/T-KAM.jl")
using .T_KAM_model

include("../src/T-KAM/model_setup.jl")
using .ModelSetup

include("../src/T-KAM/loss_fcns/thermodynamic.jl")
using .ThermodynamicIntegration: sample_thermo

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "4")
out_dim = parse(Int, retrieve(conf, "GeneratorModel", "output_dim"))

function test_posterior_sampling()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 3, 50)
    model = init_T_KAM(dataset, conf, (32, 32, 3))
    x_test = first(model.train_loader) |> pu
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

    if model.prior.bool_config.mixture_model || model.prior.bool_config.ula
        @test size(z_posterior) == (10, 1, 10, 4)
    else
        @test size(z_posterior) == (10, 5, 10, 4)
    end
    @test size(temps) == (4,)
    @test !any(isnan, z_posterior)
end

function test_model_derivative()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 1, 50)
    model = init_T_KAM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> pu
    model, ps, st_kan, st_lux = prep_model(model, x_test)
    ps = half_quant.(ps)
    ∇ = zero(half_quant) .* ps

    loss, ∇, st_ebm, st_gen =
        model.loss_fcn(ps, ∇, st_kan, st_lux, model, x_test; rng = Random.default_rng())
    @test norm(∇) != 0
    @test !any(isnan, ∇)
end

@testset "Thermodynamic Integration Tests" begin
    test_posterior_sampling()
    test_model_derivative()
end
