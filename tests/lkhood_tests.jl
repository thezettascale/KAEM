using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/T-KAM/ebm/ebm_model.jl")
include("../src/T-KAM/gen/gen_model.jl")
include("../src/utils.jl")
using .EBM_Model: init_EbmModel, EbmModel
using .GeneratorModel
using .Utils

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
out_dim = parse(Int, retrieve(conf, "GeneratorModel", "output_dim"))
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
MC_sample_size = parse(Int, retrieve(conf, "TRAINING", "importance_sample_size"))
z_dim = last(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))

Random.seed!(42)
EBM = init_EbmModel(conf)
ebm_ps, ebm_st = Lux.setup(Random.GLOBAL_RNG, EBM)

struct PriorWrapper{T<:EbmModel{Float32}}
    prior::T
end

wrap = PriorWrapper(EBM)

function test_generate()
    Random.seed!(42)
    commit!(conf, "CNN", "use_cnn_lkhood", "false")
    lkhood = init_GenModel(conf, (32, 32, 1))
    gen_ps, gen_st = Lux.setup(Random.GLOBAL_RNG, lkhood)

    ps = (ebm = ebm_ps, gen = gen_ps) |> ComponentArray |> device
    st = (ebm = ebm_st, gen = gen_st) |> device

    z = first(wrap.prior.sample_z(wrap, b_size, ps, st, default_rng()))
    x, _ = lkhood.generate_from_z(lkhood, ps.gen, st.gen, z)
    @test size(x) == (32, 32, 1, b_size)
end

function test_cnn_generate()
    Random.seed!(42)
    commit!(conf, "CNN", "use_cnn_lkhood", "true")
    lkhood = init_GenModel(conf, (32, 32, out_dim))
    gen_ps, gen_st = Lux.setup(Random.GLOBAL_RNG, lkhood)

    ps = (ebm = ebm_ps, gen = gen_ps) |> ComponentArray |> device
    st = (ebm = ebm_st, gen = gen_st) |> device

    z = first(wrap.prior.sample_z(wrap, b_size, ps, st, default_rng()))
    x, _ = lkhood.generate_from_z(lkhood, ps.gen, Lux.testmode(st.gen), z)
    @test size(x) == (32, 32, out_dim, b_size)

    commit!(conf, "CNN", "use_cnn_lkhood", "false")
end

function test_seq_generate()
    Random.seed!(42)
    commit!(conf, "SEQ", "sequence_length", "8")

    lkhood = init_GenModel(conf, (out_dim, 8))
    gen_ps, gen_st = Lux.setup(default_rng(), lkhood)

    ps = (ebm = ebm_ps, gen = gen_ps) |> ComponentArray |> device
    st = (ebm = ebm_st, gen = gen_st) |> device

    z = first(wrap.prior.sample_z(wrap, b_size, ps, st, default_rng()))
    x, _ = lkhood.generate_from_z(lkhood, ps.gen, Lux.testmode(st.gen), z)
    @test size(x) == (lkhood.out_size, 8, b_size)

    commit!(conf, "SEQ", "sequence_length", "1")
end

function test_logllhood()
    Random.seed!(42)
    lkhood = init_GenModel(conf, (out_dim, out_dim, 1))
    gen_ps, gen_st = Lux.setup(default_rng(), lkhood)

    ps = (ebm = ebm_ps, gen = gen_ps) |> ComponentArray |> device
    st = (ebm = ebm_st, gen = gen_st) |> device

    x = randn(half_quant, out_dim, out_dim, 1, b_size) |> device
    z = first(wrap.prior.sample_z(wrap, b_size, ps, st, default_rng()))
    logllhood, _, _ = log_likelihood_IS(z, x, lkhood, ps.gen, st.gen)
    @test size(logllhood) == (b_size, b_size)
end

@testset "KAN Likelihood Tests" begin
    test_generate()
    test_cnn_generate()
    test_seq_generate()
    test_logllhood()
end
