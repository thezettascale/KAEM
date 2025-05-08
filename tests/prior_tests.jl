using Test, Random, LinearAlgebra, Lux, ConfParser, Zygote

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/T-KAM/EBM_prior.jl")
include("../src/utils.jl")
using .ebm_ebm_prior: ebm_prior, init_ebm_prior, log_prior, sample_prior
using .Utils

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
p_size = first(parse.(Int, retrieve(conf, "EBM_PRIOR", "layer_widths")))
q_size = last(parse.(Int, retrieve(conf, "EBM_PRIOR", "layer_widths")))

function test_sampling()
    Random.seed!(42)
    prior = init_ebm_prior(conf; prior_seed=1) 

    @test prior.p_size == p_size
    @test prior.q_size == q_size

    ps, st = Lux.setup(Random.GLOBAL_RNG, prior)
    ps, st = ps |> device, st |> device

    z_test = sample_prior(prior, b_size, ps, st) 
    @test all(size(z_test) .== (q_size, p_size, b_size))
end

function test_log_prior()
    Random.seed!(42)
    prior = init_ebm_prior(conf; prior_seed=1) 
    ps, st = Lux.setup(Random.GLOBAL_RNG, prior)
    ps, st = ps |> device, st |> device

    z_test = sample_prior(prior, b_size, ps, st)
    log_p = first(log_prior(prior, z_test, ps, st))
    @test size(log_p) == (b_size, )
end

function test_log_prior_derivative()
    Random.seed!(42)
    prior = init_ebm_prior(conf; prior_seed=1) 
    ps, st = Lux.setup(Random.GLOBAL_RNG, prior)
    ps, st = ps |> device, st |> device

    z_test = sample_prior(prior, b_size, ps, st)
    ∇ = first(gradient(x -> sum(first(log_prior(prior, x, ps, st))), z_test))
    @test size(∇) == size(z_test)
end
    

@testset "Mixture Prior Tests" begin
    test_sampling()
    test_log_prior()
    test_log_prior_derivative()
end