using Test, Random, LinearAlgebra, Statistics, Lux, ConfParser, ComponentArrays, Reactant

ENV["GPU"] = true

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
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
p_size = first(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))
q_size = last(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))

rng = Random.MersenneTwister(1)
dataset = randn(rng, Float32, 32, 32, 1, b_size * 10)
model = init_KAEM(dataset, conf, (32, 32, 1))
x_test = first(model.train_loader) |> pu
optimizer = create_opt(conf)
model, _, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; MLIR = false)

function test_shapes()
    @test model.prior.p_size == p_size
    return @test model.prior.q_size == q_size
end

function test_uniform_prior()
    commit!(conf, "EbmModel", "π_0", "uniform")

    compiled_sample = Reactant.@compile model.sample_prior(model, ps, st_kan, st_lux, st_rng)
    z_test = first(compiled_sample(model, ps, st_kan, st_lux, st_rng))

    if model.prior.bool_config.mixture_model || model.prior.bool_config.ula
        @test size(z_test) == (q_size, 1, b_size)
    else
        @test size(z_test) == (q_size, p_size, b_size)
    end

    compiled_log_prior = Reactant.@compile model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad)
    log_p = first(compiled_log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad))

    @test !any(isnan, Array(z_test))
    @test size(log_p) == (b_size,)
    return @test !any(isnan, Array(log_p))
end

function test_gaussian_prior()
    commit!(conf, "EbmModel", "π_0", "gaussian")
    compiled_sample = Reactant.@compile model.sample_prior(model, ps, st_kan, st_lux, st_rng)
    z_test = first(compiled_sample(model, ps, st_kan, st_lux, st_rng))

    if model.prior.bool_config.mixture_model || model.prior.bool_config.ula
        @test all(size(z_test) .== (q_size, 1, b_size))
    else
        @test all(size(z_test) .== (q_size, p_size, b_size))
    end

    compiled_log_prior = Reactant.@compile model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad)
    log_p = first(compiled_log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad))

    @test !any(isnan, Array(z_test))
    @test size(log_p) == (b_size,)
    @test !any(isnan, Array(log_p))

    z_array = Array(z_test)
    z_flat = reshape(z_array, :)

    @test abs(mean(z_flat)) < 2.0f0  # Roughly centered
    @test std(z_flat) > 1.0f-2  # Not degenerate
    @test all(Array(log_p) .> -1.0f6)  # Not extreme
    return @test all(Array(log_p) .< 1.0f6)
end

function test_lognormal_prior()
    commit!(conf, "EbmModel", "π_0", "lognormal")
    compiled_sample = Reactant.@compile model.sample_prior(model, ps, st_kan, st_lux, st_rng)
    z_test = first(compiled_sample(model, ps, st_kan, st_lux, st_rng))

    if model.prior.bool_config.mixture_model || model.prior.bool_config.ula
        @test all(size(z_test) .== (q_size, 1, b_size))
    else
        @test all(size(z_test) .== (q_size, p_size, b_size))
    end

    compiled_log_prior = Reactant.@compile model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad)
    log_p = first(compiled_log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad))

    @test !any(isnan, Array(z_test))
    @test size(log_p) == (b_size,)
    return @test !any(isnan, Array(log_p))
end

function test_learnable_gaussian_prior()
    commit!(conf, "EbmModel", "π_0", "learnable_gaussian")
    compiled_sample = Reactant.@compile model.sample_prior(model, ps, st_kan, st_lux, st_rng)
    z_test = first(compiled_sample(model, ps, st_kan, st_lux, st_rng))

    if model.prior.bool_config.mixture_model || model.prior.bool_config.ula
        @test all(size(z_test) .== (q_size, 1, b_size))
    else
        @test all(size(z_test) .== (q_size, p_size, b_size))
    end

    compiled_log_prior = Reactant.@compile model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad)
    log_p = first(compiled_log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad))

    @test !any(isnan, Array(z_test))
    @test size(log_p) == (b_size,)
    return @test !any(isnan, Array(log_p))
end

function test_ebm_prior()
    commit!(conf, "EbmModel", "π_0", "ebm")
    compiled_sample = Reactant.@compile model.sample_prior(model, ps, st_kan, st_lux, st_rng)
    z_test = first(compiled_sample(model, ps, st_kan, st_lux, st_rng))

    if model.prior.bool_config.mixture_model || model.prior.bool_config.ula
        @test all(size(z_test) .== (q_size, 1, b_size))
    else
        @test all(size(z_test) .== (q_size, p_size, b_size))
    end

    compiled_log_prior = Reactant.@compile model.log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad)
    log_p = first(compiled_log_prior(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm, st_kan.quad))

    @test !any(isnan, Array(z_test))
    @test size(log_p) == (b_size,)
    return @test !any(isnan, Array(log_p))
end

@testset "Mixture Prior Tests" begin
    test_shapes()
    test_uniform_prior()
    test_gaussian_prior()
    test_lognormal_prior()
    test_learnable_gaussian_prior()
    test_ebm_prior()
end
