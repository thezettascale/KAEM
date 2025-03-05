using Test, Random, LinearAlgebra, Lux, ConfParser, Zygote, ComponentArrays

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../T-KAM/ebm_prior.jl")
include("../T-KAM/KAN_likelihood.jl")
include("../utils.jl")
using .ebm_ebm_prior
using .KAN_likelihood
using .Utils

conf = ConfParse("src/tests/test_conf.ini")
parse_conf!(conf)
out_dim = parse(Int, retrieve(conf, "KAN_LIKELIHOOD", "output_dim"))
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
MC_sample_size = parse(Int, retrieve(conf, "TRAINING", "importance_sample_size"))
z_dim = last(parse.(Int, retrieve(conf, "EBM_PRIOR", "layer_widths")))

function test_generate()
    Random.seed!(42)
    lkhood = init_KAN_lkhood(conf, (out_dim, out_dim, 1); lkhood_seed=1)
    gen_ps, gen_st = Lux.setup(Random.GLOBAL_RNG, lkhood)

    prior = init_ebm_prior(conf; prior_seed=1)
    ebm_ps, ebm_st = Lux.setup(Random.GLOBAL_RNG, prior)

    ps = (ebm=ebm_ps, gen=gen_ps) |> device
    st = (ebm=ebm_st, gen=gen_st) |> device

    z = first(prior.sample_z(prior, b_size, ps.ebm, st.ebm, 1))
    x, _ = lkhood.generate_from_z(lkhood, ps.gen, st.gen, z)
    @test size(x) == (out_dim, out_dim, 1, b_size)
end

function test_cnn_generate()
    Random.seed!(42)
    commit!(conf, "CNN", "use_cnn_lkhood", "true")

    lkhood = init_KAN_lkhood(conf, (32, 32, out_dim); lkhood_seed=1)
    gen_ps, gen_st = Lux.setup(Random.GLOBAL_RNG, lkhood)

    prior = init_ebm_prior(conf; prior_seed=1)
    ebm_ps, ebm_st = Lux.setup(Random.GLOBAL_RNG, prior)

    ps = (ebm=ebm_ps, gen=gen_ps) |> device
    st = (ebm=ebm_st, gen=gen_st) |> device

    z = first(prior.sample_z(prior, b_size, ps.ebm, st.ebm, 1))
    x, _ = lkhood.generate_from_z(lkhood, ps.gen, Lux.testmode(st.gen), z)
    @test size(x) == (32, 32, out_dim, b_size)

    commit!(conf, "CNN", "use_cnn_lkhood", "false")
end

function test_SEQ_generate()
    Random.seed!(42)
    commit!(conf, "SEQ", "sequence_length", "8")

    lkhood = init_KAN_lkhood(conf, (out_dim, 8); lkhood_seed=1)
    gen_ps, gen_st = Lux.setup(Random.GLOBAL_RNG, lkhood)

    prior = init_ebm_prior(conf; prior_seed=1)
    ebm_ps, ebm_st = Lux.setup(Random.GLOBAL_RNG, prior)

    ps = (ebm=ebm_ps, gen=gen_ps) |> device
    st = (ebm=ebm_st, gen=gen_st) |> device 

    z = first(prior.sample_z(prior, b_size, ps.ebm, st.ebm, 1))
    x, _ = lkhood.generate_from_z(lkhood, ps.gen, Lux.testmode(st.gen), z)
    @test size(x) == (lkhood.out_size, 8, b_size)

    commit!(conf, "SEQ", "sequence_length", "1")
end

function test_logllhood()
    Random.seed!(42)
    lkhood = init_KAN_lkhood(conf, (out_dim, out_dim, 1); lkhood_seed=1)
    gen_ps, gen_st = Lux.setup(Random.GLOBAL_RNG, lkhood)

    prior = init_ebm_prior(conf; prior_seed=1)
    ebm_ps, ebm_st = Lux.setup(Random.GLOBAL_RNG, prior)

    ps = (ebm=ebm_ps, gen=gen_ps) |> device
    st = (ebm=ebm_st, gen=gen_st) |> device

    x = randn(Float32, out_dim, out_dim, 1, b_size) |> device
    z = first(prior.sample_z(prior, b_size, ps.ebm, st.ebm, 1))
    logllhood, _, _ = log_likelihood(lkhood, ps.gen, st.gen, x, z)
    @test size(logllhood) == (b_size, b_size)
end

function test_derivative()
    Random.seed!(42)
    lkhood = init_KAN_lkhood(conf, (out_dim, out_dim, 1); lkhood_seed=1)
    gen_ps, gen_st = Lux.setup(Random.GLOBAL_RNG, lkhood)

    prior = init_ebm_prior(conf; prior_seed=1)
    ebm_ps, ebm_st = Lux.setup(Random.GLOBAL_RNG, prior)

    ps = (ebm=ebm_ps, gen=gen_ps) |> device
    st = (ebm=ebm_st, gen=gen_st) |> device

    x = randn(Float32, out_dim, out_dim, 1, b_size) |> device
    z = first(prior.sample_z(prior, b_size, ps.ebm, st.ebm, 1))
    ∇ = first(gradient(z_i -> sum(first(log_likelihood(lkhood, ps.gen, st.gen, x, z_i))), z))
    @test size(∇) == size(z)
end

function test_cnn_derivative()
    Random.seed!(42)
    commit!(conf, "CNN", "use_cnn_lkhood", "true")

    lkhood = init_KAN_lkhood(conf, (32, 32, out_dim); lkhood_seed=1)
    gen_ps, gen_st = Lux.setup(Random.GLOBAL_RNG, lkhood)

    prior = init_ebm_prior(conf; prior_seed=1)
    ebm_ps, ebm_st = Lux.setup(Random.GLOBAL_RNG, prior)

    ps = ComponentArray(ebm=ebm_ps, gen=gen_ps) |> device
    st = (ebm=ebm_st, gen=gen_st) |> device

    x = randn(Float32, 32, 32, out_dim, b_size) |> device
    z = first(prior.sample_z(prior, b_size, ps.ebm, st.ebm, 1))
    ∇ = first(gradient(p -> sum(first(log_likelihood(lkhood, p, st.gen, x, z))), ps.gen))
    @test size(∇) == size(ps.gen)

    commit!(conf, "CNN", "use_cnn_lkhood", "false")
end

function test_SEQ_derivative()
    Random.seed!(42)
    commit!(conf, "SEQ", "sequence_length", "8")

    lkhood = init_KAN_lkhood(conf, (out_dim, 8); lkhood_seed=1)
    gen_ps, gen_st = Lux.setup(Random.GLOBAL_RNG, lkhood)

    prior = init_ebm_prior(conf; prior_seed=1)
    ebm_ps, ebm_st = Lux.setup(Random.GLOBAL_RNG, prior)

    ps = ComponentArray(ebm=ebm_ps, gen=gen_ps) |> device
    st = (ebm=ebm_st, gen=gen_st) |> device 

    x = randn(Float32, lkhood.out_size, 8, b_size) |> device
    z = first(prior.sample_z(prior, b_size, ps.ebm, st.ebm, 1))
    ∇ = first(gradient(p -> sum(first(log_likelihood(lkhood, p, st.gen, x, z))), ps.gen))
    @test size(∇) == size(ps.gen)

    commit!(conf, "SEQ", "sequence_length", "1")
end

@testset "KAN Likelihood Tests" begin
    test_generate()
    test_cnn_generate()
    test_SEQ_generate()
    test_logllhood()
    test_derivative()
    test_cnn_derivative()
    test_SEQ_derivative()
end