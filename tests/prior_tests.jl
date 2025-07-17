using Test, Random, LinearAlgebra, Lux, ConfParser, Enzyme, ComponentArrays

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

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

ps = (ebm = ps, gen = ps) |> ComponentArray |> device
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
    log_p = first(wrap.prior.lp_fcn(z_test, wrap.prior, ps.ebm, st.ebm))
    @test size(log_p) == (b_size,)
end

function test_log_prior_derivative()
    z_test = first(wrap.prior.sample_z(wrap, b_size, ps, st, 42))
    ∇ = zeros(half_quant, size(z_test)) |> device

    f = (x, p, s, ebm) -> sum(first(wrap.prior.lp_fcn(x, ebm, p, s)))
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        f,
        Enzyme.Active,
        Enzyme.Duplicated(z_test, ∇),
        Enzyme.Const(ps.ebm),
        Enzyme.Const(st.ebm),
        Enzyme.Const(wrap.prior),
    )

    @test size(∇) == size(z_test)
end

@testset "Mixture Prior Tests" begin
    test_sampling()
    test_log_prior()
    # test_log_prior_derivative()
end
