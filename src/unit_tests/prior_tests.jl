using Test, Random, LinearAlgebra, Lux, ConfParser, Zygote

ENV["GPU"] = true

include("../LV-KAM/mixture_prior.jl")
include("../utils.jl")
using .ebm_mix_prior
using .Utils

conf = ConfParse("src/unit_tests/test_conf.ini")
parse_conf!(conf)
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
MC_batch_size = parse(Int, retrieve(conf, "TRAINING", "MC_estimate_subbatch_size"))

function test_sampling()
    Random.seed!(42)
    prior = init_mix_prior(conf; prior_seed=1) 
    ps, st = Lux.setup(Random.GLOBAL_RNG, prior)
    ps, st = ps |> device, st |> device

    z_test = first(sample_prior(prior, b_size, ps, st))
    @test all(size(z_test) .== (b_size, parse(Int, retrieve(conf, "MIX_PRIOR", "hidden_dim"))))
end

function test_log_prior()
    Random.seed!(42)
    prior = init_mix_prior(conf; prior_seed=1) 
    ps, st = Lux.setup(Random.GLOBAL_RNG, prior)
    ps, st = ps |> device, st |> device

    z_test = first(sample_prior(prior, b_size*prior.num_latent_samples, ps, st))
    log_p = log_prior(prior, z_test, ps, st)
    @test size(log_p) == (b_size, 1, prior.num_latent_samples)
end

function test_log_prior_derivative()
    Random.seed!(42)
    prior = init_mix_prior(conf; prior_seed=1) 
    ps, st = Lux.setup(Random.GLOBAL_RNG, prior)
    ps, st = ps |> device, st |> device

    z_test = first(sample_prior(prior, b_size*prior.num_latent_samples, ps, st))
    ∇ = first(gradient(x -> sum(log_prior(prior, x, ps, st)), z_test))
    @test size(∇) == size(z_test)
end
    

@testset "Mixture Prior Tests" begin
    test_sampling()
    test_log_prior()
    test_log_prior_derivative()
end