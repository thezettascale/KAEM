using Test, Random, LinearAlgebra, Lux, ConfParser

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

@testset "Mixture Prior Tests" begin
    test_sampling()
end