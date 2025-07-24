using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays, CUDA


ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/T-KAM/T-KAM.jl")
include("../src/T-KAM/model_setup.jl")
include("../src/utils.jl")
include("../src/T-KAM/gen/loglikelihoods.jl")
using .T_KAM_model
using .ModelSetup
using .Utils: device, half_quant, full_quant
using .LogLikelihoods: log_likelihood_IS, log_likelihood_MALA

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
out_dim = parse(Int, retrieve(conf, "GeneratorModel", "output_dim"))
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
MC_sample_size = parse(Int, retrieve(conf, "TRAINING", "importance_sample_size"))
z_dim = last(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))

Random.seed!(42)

function test_generate()
    Random.seed!(42)
    commit!(conf, "CNN", "use_cnn_lkhood", "false")
    dataset = randn(full_quant, 32, 32, 1, 50)
    model = init_T_KAM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> device
    model, ps, st_kan, st_lux = prep_model(model, x_test)
    ps = half_quant.(ps)

    z = first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))
    x, _ = model.lkhood.generator(ps.gen, st_kan.gen, st_lux.gen, z)
    @test size(x) == (32, 32, 1, b_size)
end

function test_logllhood()
    Random.seed!(42)
    dataset = randn(full_quant, 32, 32, 1, 50)
    model = init_T_KAM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> device
    model, ps, st_kan, st_lux = prep_model(model, x_test)
    ps = half_quant.(ps)

    x = randn(half_quant, 32, 32, 1, b_size) |> device
    z = first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))
    noise = randn(half_quant, 32, 32, 1, b_size, b_size) |> device
    logllhood, _ =
        log_likelihood_IS(z, x, model.lkhood, ps.gen, st_kan.gen, st_lux.gen, noise)
    @test size(logllhood) == (b_size, b_size)
end

function test_cnn_generate()
    Random.seed!(42)
    commit!(conf, "CNN", "use_cnn_lkhood", "true")
    dataset = randn(full_quant, 32, 32, out_dim, 50)
    model = init_T_KAM(dataset, conf, (32, 32, out_dim))
    x_test = first(model.train_loader) |> device
    model, ps, st_kan, st_lux = prep_model(model, x_test)
    ps = half_quant.(ps)

    z = first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))
    x = first(model.lkhood.generator(ps.gen, st_kan.gen, st_lux.gen, z))
    @test size(x) == (32, 32, out_dim, b_size)

    commit!(conf, "CNN", "use_cnn_lkhood", "false")
end

function test_seq_generate()
    Random.seed!(42)
    commit!(conf, "SEQ", "sequence_length", "8")

    dataset = randn(full_quant, out_dim, 8, 50)
    model = init_T_KAM(dataset, conf, (out_dim, 8))
    x_test = first(model.train_loader) |> device
    model, ps, st_kan, st_lux = prep_model(model, x_test)
    ps = half_quant.(ps)

    z = first(model.sample_prior(model, b_size, ps, st_kan, st_lux, Random.default_rng()))
    x, _ = model.lkhood.generator(ps.gen, st_kan.gen, st_lux.gen, z)
    @test size(x) == (out_dim, 8, b_size)

    commit!(conf, "SEQ", "sequence_length", "1")
end

@testset "KAN Likelihood Tests" begin
    test_generate()
    test_logllhood()
    test_cnn_generate()
    test_seq_generate()
end
