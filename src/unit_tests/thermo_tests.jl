using Test, Random, LinearAlgebra, Lux, ConfParser, Zygote, ComponentArrays

ENV["GPU"] = true

include("../LV-KAM/LV-KAM.jl")
include("../LV-KAM/thermodynamic_integration.jl")
include("../utils.jl")
using .LV_KAM_model
using .ThermodynamicIntegration
using .Utils

conf = ConfParse("src/unit_tests/thermo_config.ini")
parse_conf!(conf)
out_dim = parse(Int, retrieve(conf, "MOE_LIKELIHOOD", "output_dim"))

function test_sample_prior()
    Random.seed!(42)
    dataset = randn(Float32, 3, 50) 
    m = init_LV_KAM(dataset, conf)
    ps, st = Lux.setup(Random.GLOBAL_RNG, m)
    ps, st = ComponentArray(ps) |> device, st |> device

    z, seed = z, seed = m.prior.sample_z(
        m.prior, 
        m.MC_samples,
        ps.ebm,
        st.ebm,
        seed
        )

    @test all(size(z) .== (m.MC_samples, m.lkhood.Λ_fcns[Symbol("1")].in_dim))
end

function test_sample_prior_derivative()
    Random.seed!(42)
    dataset = randn(Float32, 3, 50) 
    m = init_LV_KAM(dataset, conf)
    ps, st = Lux.setup(Random.GLOBAL_RNG, m)
    ps, st = ComponentArray(ps) |> device, st |> device

    fcn = p -> sum(first(m.prior.sample_z(
        m.prior, 
        m.MC_samples,
        p.ebm,
        st.ebm,
        seed
        )))

    ∇ = first(gradient(p -> fcn(p), ps))
    @test norm(∇) > 0
    @test !any(isnan, ∇)
end
    
function test_model_derivative()
    Random.seed!(42)
    dataset = randn(Float32, 3, 50) 
    model = init_LV_KAM(dataset, conf)
    x_test = first(model.train_loader)' |> device
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    ps, st = ComponentArray(ps) |> device, st |> device

    ∇ = first(gradient(p -> TI_loss(model, p, st, x_test), ps))
    @test norm(∇) > 0
    @test !any(isnan, ∇)
end

@testset "Thermodynamic Integration Tests" begin
    test_sample_prior()
    test_sample_prior_derivative()
    test_model_derivative()
end