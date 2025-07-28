using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays, CUDA

ENV["GPU"] = false
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/utils.jl")
using .Utils

include("../src/T-KAM/T-KAM.jl")
using .T_KAM_model

include("../src/T-KAM/model_setup.jl")
using .ModelSetup

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
p_size = first(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))
q_size = last(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))

Random.seed!(42)
dataset = randn(full_quant, 32, 32, 1, b_size*10)
model = init_T_KAM(dataset, conf, (32, 32, 1))
x_test = first(model.train_loader) |> pu
model, ps, st_kan, st_lux = prep_model(model, x_test)
ps = half_quant.(ps)

function test_shapes()
    @test model.prior.p_size == p_size
    @test model.prior.q_size == q_size
end

function test_uniform_prior()
    commit!(conf, "EbmModel", "π_0", "uniform")
    z_test =
        first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))

    if model.prior.mixture_model || model.prior.ula
        @test all(size(z_test) .== (q_size, 1, b_size))
    else
        @test all(size(z_test) .== (q_size, p_size, b_size))
    end

    log_p = first(model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm))

    @test !any(isnan, z_test)
    @test size(log_p) == (b_size,)
    @test !any(isnan, log_p)
end

function test_gaussian_prior()
    commit!(conf, "EbmModel", "π_0", "gaussian")
    z_test =
        first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))

    if model.prior.mixture_model || model.prior.ula
        @test all(size(z_test) .== (q_size, 1, b_size))
    else
        @test all(size(z_test) .== (q_size, p_size, b_size))
    end

    log_p = first(model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm))

    @test !any(isnan, z_test)
    @test size(log_p) == (b_size,)
    @test !any(isnan, log_p)
end

function test_lognormal_prior()
    commit!(conf, "EbmModel", "π_0", "lognormal")
    z_test =
        first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))

    if model.prior.mixture_model || model.prior.ula
        @test all(size(z_test) .== (q_size, 1, b_size))
    else
        @test all(size(z_test) .== (q_size, p_size, b_size))
    end

    log_p = first(model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm))

    @test !any(isnan, z_test)
    @test size(log_p) == (b_size,)
    @test !any(isnan, log_p)
end

function test_learnable_gaussian_prior()
    commit!(conf, "EbmModel", "π_0", "learnable_gaussian")
    z_test =
        first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))

    if model.prior.mixture_model || model.prior.ula
        @test all(size(z_test) .== (q_size, 1, b_size))
    else
        @test all(size(z_test) .== (q_size, p_size, b_size))
    end

    @test !any(isnan, z_test)
    @test size(log_p) == (b_size,)
    @test !any(isnan, log_p)
end

function test_ebm_prior()
    commit!(conf, "EbmModel", "π_0", "ebm")
    z_test =
        first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))

    if model.prior.mixture_model || model.prior.ula
        @test all(size(z_test) .== (q_size, 1, b_size))
    else
        @test all(size(z_test) .== (q_size, p_size, b_size))
    end

    log_p = first(model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm))

    @test !any(isnan, z_test)
    @test size(log_p) == (b_size,)
    @test !any(isnan, log_p)
end

@testset "Mixture Prior Tests" begin
    test_uniform_prior()
    test_gaussian_prior()
    test_lognormal_prior()
    test_learnable_gaussian_prior()
    test_ebm_prior()
end
