using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays, CUDA

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"


include("../src/T-KAM/T-KAM.jl")
include("../src/T-KAM/model_setup.jl")
include("../src/utils.jl")
using .T_KAM_model
using .ModelSetup: prep_model
using .Utils: device, half_quant, full_quant

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
p_size = first(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))
q_size = last(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))

Random.seed!(42)
dataset = randn(full_quant, 32, 32, 1, b_size*10)
model = init_T_KAM(dataset, conf, (32, 32, 1))
x_test = first(model.train_loader) |> device
model, ps, st_kan, st_lux = prep_model(model, x_test)
ps = half_quant.(ps)

function test_shapes()
    @test model.prior.p_size == p_size
    @test model.prior.q_size == q_size
end

function test_sampling()
    z_test =
        first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))
    @test all(size(z_test) .== (q_size, p_size, b_size))
end

function test_log_prior()
    z_test =
        first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))
    log_p = first(model.prior.lp_fcn(z_test, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm))
    @test size(log_p) == (b_size,)
end

@testset "Mixture Prior Tests" begin
    test_sampling()
    test_log_prior()
end
