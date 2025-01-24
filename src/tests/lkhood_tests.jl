using Test, Random, LinearAlgebra, Lux, ConfParser, Zygote

ENV["GPU"] = true
ENV["QUANT"] = "FP32"

include("../T-KAM/mixture_prior.jl")
include("../T-KAM/KAN_likelihood.jl")
include("../utils.jl")
using .ebm_mix_prior
using .KAN_likelihood
using .Utils

conf = ConfParse("src/tests/test_conf.ini")
parse_conf!(conf)
out_dim = parse(Int, retrieve(conf, "KAN_LIKELIHOOD", "output_dim"))
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
MC_sample_size = parse(Int, retrieve(conf, "TRAINING", "importance_sample_size"))
z_dim = last(parse.(Int, retrieve(conf, "MIX_PRIOR", "layer_widths")))

function test_generate()
    Random.seed!(42)
    lkhood = init_KAN_lkhood(conf, out_dim; lkhood_seed=1)
    gen_ps, gen_st = Lux.setup(Random.GLOBAL_RNG, lkhood)

    prior = init_mix_prior(conf; prior_seed=1)
    ebm_ps, ebm_st = Lux.setup(Random.GLOBAL_RNG, prior)

    ps = (ebm=ebm_ps, gen=gen_ps) |> device
    st = (ebm=ebm_st, gen=gen_st) |> device

    z, seed = prior.sample_z(prior, b_size, ps.ebm, st.ebm, 1)
    x = generate_from_z(lkhood, ps.gen, st.gen, z)
    @test size(x) == (b_size, out_dim)
end

function test_logllhood()
    Random.seed!(42)
    lkhood = init_KAN_lkhood(conf, out_dim; lkhood_seed=1)
    gen_ps, gen_st = Lux.setup(Random.GLOBAL_RNG, lkhood)

    prior = init_mix_prior(conf; prior_seed=1)
    ebm_ps, ebm_st = Lux.setup(Random.GLOBAL_RNG, prior)

    ps = (ebm=ebm_ps, gen=gen_ps) |> device
    st = (ebm=ebm_st, gen=gen_st) |> device

    x = randn(Float32, out_dim, b_size) |> device
    z, seed = prior.sample_z(prior, b_size, ps.ebm, st.ebm, 1)
    logllhood, _ = log_likelihood(lkhood, ps.gen, st.gen, x, z)
    @test size(logllhood) == (b_size,b_size)
end

@testset "KAN Likelihood Tests" begin
    test_generate()
    test_logllhood()
end