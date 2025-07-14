using Test, Random, LinearAlgebra, Lux, ConfParser, DifferentiationInterface

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"
ENV["AD_BACKEND"] = "ENZYME"

include("../src/T-KAM/ebm/ebm_model.jl")
include("../src/utils.jl")
using .EBM_Model: EbmModel, init_EbmModel
using .Utils

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
p_size = first(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))
q_size = last(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))

Random.seed!(42)
EBM = init_EbmModel(conf; prior_seed = 1)
ps, st = Lux.setup(Random.GLOBAL_RNG, EBM)

struct PriorWrapper{T<:EbmModel{Float32}}
    prior::T
end

wrap = PriorWrapper(EBM)

ps = (ebm = ps, gen = ps) |> device
st = (ebm = st, gen = st) |> device

function test_shapes()
    @test prior.p_size == p_size
    @test prior.q_size == q_size
end

function test_sampling()
    z_test = first(wrap.prior.sample_z(wrap, b_size, ps, st, 42))
    @test all(size(z_test) .== (q_size, p_size, b_size))
end

function test_log_prior()
    z_test = first(wrap.prior.sample_z(wrap, b_size, ps, st, 42))
    log_p = first(wrap.prior.lp_fcn(wrap.prior, z_test, ps.ebm, st.ebm))
    @test size(log_p) == (b_size,)
end

function test_log_prior_derivative()
    z_test = first(wrap.prior.sample_z(wrap, b_size, ps, st, 42))
    ∇ = gradient(
        x -> sum(first(wrap.prior.lp_fcn(wrap.prior, x, ps.ebm, st.ebm))),
        AD_backend,
        z_test,
    )
    @test size(∇) == size(z_test)
end

@testset "Mixture Prior Tests" begin
    test_sampling()
    test_log_prior()
    test_log_prior_derivative()
end
