using Test, Random, LinearAlgebra, Lux, ConfParser, Zygote

ENV["GPU"] = true

include("../LV-KAM/mixture_prior.jl")
include("../utils.jl")
using .ebm_mix_prior
using .Utils

conf = ConfParse("src/tests/test_conf.ini")
parse_conf!(conf)
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
z_dim = last(parse.(Int, retrieve(conf, "MIX_PRIOR", "layer_widths")))

function test_sampling()
    Random.seed!(42)
    prior = init_mix_prior(conf; prior_seed=1) 
    ps, st = Lux.setup(Random.GLOBAL_RNG, prior)
    ps, st = ps |> device, st |> device

    z_test = first(prior.sample_z(prior, b_size, ps, st,1))
    @test all(size(z_test) .== (b_size, z_dim))
end

function test_log_prior()
    Random.seed!(42)
    prior = init_mix_prior(conf; prior_seed=1) 
    ps, st = Lux.setup(Random.GLOBAL_RNG, prior)
    ps, st = ps |> device, st |> device

    z_test = first(prior.sample_z(prior, b_size, ps, st, 1))
    log_p = log_prior(prior, z_test, ps, st)
    @test size(log_p) == (b_size, 1)
end

function test_log_prior_derivative()
    Random.seed!(42)
    prior = init_mix_prior(conf; prior_seed=1) 
    ps, st = Lux.setup(Random.GLOBAL_RNG, prior)
    ps, st = ps |> device, st |> device

    z_test = first(prior.sample_z(prior, b_size, ps, st, 1))
    ∇ = first(gradient(x -> sum(log_prior(prior, x, ps, st)), z_test))
    @test size(∇) == size(z_test)
end
    

@testset "Mixture Prior Tests" begin
    test_sampling()
    test_log_prior()
    test_log_prior_derivative()
end