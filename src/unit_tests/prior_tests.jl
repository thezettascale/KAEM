using Test, Random, LinearAlgebra, Lux, ConfParser, Zygote

ENV["GPU"] = true

include("../LV-KAM/mixture_prior.jl")
include("../utils.jl")
using .ebm_mix_prior
using .Utils

conf = ConfParse("src/unit_tests/test_conf.ini")
parse_conf!(conf)

function test_sampling()
    Random.seed!(42)
    prior = init_mix_prior(conf; prior_seed=1) 
    ps, st = Lux.setup(Random.GLOBAL_RNG, prior)
    ps, st = ps |> device, st |> device

    z_test = first(sample_prior(prior, 5, ps, st))
    @test all(size(z_test) .== (5, parse(Int, retrieve(conf, "MIX_PRIOR", "hidden_dim"))))
end

function test_log_prior()
    Random.seed!(42)
    prior = init_mix_prior(conf; prior_seed=1) 
    ps, st = Lux.setup(Random.GLOBAL_RNG, prior)
    ps, st = ps |> device, st |> device

    z_test = first(sample_prior(prior, 5, ps, st))
    log_p = log_prior(prior, z_test, ps, st)
    @test size(log_p) == (5,)
end

function test_log_prior_derivative()
    Random.seed!(42)
    prior = init_mix_prior(conf; prior_seed=1) 
    ps, st = Lux.setup(Random.GLOBAL_RNG, prior)
    ps, st = ps |> device, st |> device

    z_test = first(sample_prior(prior, 5, ps, st))
    ∇ = first(gradient(x -> sum(log_prior(prior, x, ps, st)), z_test))
    @test size(∇) == size(z_test)
end

function test_expected_prior()
    Random.seed!(42)
    prior = init_mix_prior(conf; prior_seed=1)
    ps, st = Lux.setup(Random.GLOBAL_RNG, prior)
    ps, st = ps |> device, st |> device

    func = (z, p) -> log_prior(prior, z, p, st)
    expected_p = expected_prior(prior, 5, ps, st, func)
    @test length(expected_p) == 1
end

function test_ps_derivative()
    Random.seed!(42)
    prior = init_mix_prior(conf; prior_seed=1)
    ps, st = Lux.setup(Random.GLOBAL_RNG, prior)
    ps, st = ps |> device, st |> device

    func = (z, p) -> log_prior(prior, z, p, st)
    ∇ = first(gradient(p -> expected_prior(prior, 5, p, st, func), ps))
    @test norm(∇) > 0
end
    

@testset "Mixture Prior Tests" begin
    test_sampling()
    test_log_prior()
    test_log_prior_derivative()
    test_expected_prior()
    test_ps_derivative()
end