using Test, Random, LinearAlgebra, Lux, ConfParser, Zygote

ENV["GPU"] = true

include("../LV-KAM/mixture_prior.jl")
include("../LV-KAM/MoE_likelihood.jl")
include("../utils.jl")
using .ebm_mix_prior
using .MoE_likelihood
using .Utils

conf = ConfParse("src/unit_tests/test_conf.ini")
parse_conf!(conf)
out_dim = parse(Int, retrieve(conf, "MOE_LIKELIHOOD", "output_dim"))
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
MC_batch_size = parse(Int, retrieve(conf, "TRAINING", "MC_estimate_subbatch_size"))
num_latent_samples = parse(Int, retrieve(conf, "MIX_PRIOR", "num_latent_samples"))

function test_log_likelihood()
    Random.seed!(42)
    lkhood = init_MoE_lkhood(conf, out_dim; lkhood_seed=1)
    ps, st = Lux.setup(Random.GLOBAL_RNG, lkhood)
    ps, st = ps |> device, st |> device

    z_test = randn(Float32, b_size*num_latent_samples, parse(Int, retrieve(conf, "MIX_PRIOR", "hidden_dim"))) |> device
    x_test = randn(Float32, b_size, out_dim) |> device

    log_lkhood = log_likelihood(lkhood, ps, st, x_test, z_test)
    @test size(log_lkhood) == (b_size, 1, num_latent_samples)
end

function test_log_likelihood_derivative()
    Random.seed!(42)
    lkhood = init_MoE_lkhood(conf, out_dim; lkhood_seed=1)
    ps, st = Lux.setup(Random.GLOBAL_RNG, lkhood)
    ps, st = ps |> device, st |> device

    z_test = randn(Float32, b_size, parse(Int, retrieve(conf, "MIX_PRIOR", "hidden_dim"))) |> device
    x_test = randn(Float32, b_size, out_dim) |> device
    
    ∇ = first(gradient(z -> sum(log_likelihood(lkhood, ps, st, x_test, z)), z_test))
    @test size(∇) == size(z_test)
end

function test_expected_posterior()
    Random.seed!(42)
    lkhood = init_MoE_lkhood(conf, out_dim; lkhood_seed=1)
    gen_ps, gen_st = Lux.setup(Random.GLOBAL_RNG, lkhood)
    
    prior = init_mix_prior(conf; prior_seed=1)
    ebm_ps, ebm_st = Lux.setup(Random.GLOBAL_RNG, prior)

    ps = (ebm=ebm_ps, gen=gen_ps) |> device
    st = (ebm=ebm_st, gen=gen_st) |> device

    x_test = randn(Float32, b_size, out_dim) |> device
    
    func = (z, x_i, p) -> log_likelihood(lkhood, p, st.gen, x_i, z)
    expected_p = first(expected_posterior(prior, lkhood, ps, st, x_test, func, ps.gen))
    @test length(expected_p) == b_size
end

function test_generate()
    Random.seed!(42)
    lkhood = init_MoE_lkhood(conf, out_dim; lkhood_seed=1)
    gen_ps, gen_st = Lux.setup(Random.GLOBAL_RNG, lkhood)

    prior = init_mix_prior(conf; prior_seed=1)
    ebm_ps, ebm_st = Lux.setup(Random.GLOBAL_RNG, prior)

    ps = (ebm=ebm_ps, gen=gen_ps) |> device
    st = (ebm=ebm_st, gen=gen_st) |> device

    z, seed = sample_prior(prior, b_size, ps.ebm, st.ebm)
    x = first(generate_from_z(lkhood, ps.gen, st.gen, z))
    @test size(x) == (b_size, out_dim)
end

@testset "MoE Likelihood Tests" begin
    test_log_likelihood()
    test_log_likelihood_derivative()
    test_expected_posterior()
    test_generate()
end