using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays, Reactant

ENV["GPU"] = true

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .KAEM_model

include("../src/KAEM/model_setup.jl")
using .ModelSetup

include("../src/KAEM/gen/loglikelihoods.jl")
using .LogLikelihoods

include("../src/pipeline/optimizer.jl")
using .optimization

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
out_dim = parse(Int, retrieve(conf, "GeneratorModel", "output_dim"))
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
z_dim = last(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))

rng = Random.MersenneTwister(1)
optimizer = create_opt(conf)

function test_generate()
    Random.seed!(42)
    commit!(conf, "CNN", "use_cnn_lkhood", "false")
    dataset = randn(rng, Float32, 32, 32, 1, 500)
    model = init_KAEM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> pu
    model, _, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; MLIR = false, rng = rng)

    compiled_sample_prior = Reactant.@compile model.sample_prior(model, ps, st_kan, st_lux, st_rng)
    z = first(compiled_sample_prior(model, ps, st_kan, st_lux, st_rng))
    compiled_generator = Reactant.@compile model.lkhood.generator(ps.gen, st_kan.gen, st_lux.gen, z)
    x, _ = compiled_generator(ps.gen, st_kan.gen, st_lux.gen, z)
    @test size(x) == (32, 32, 1, b_size)
    return @test !any(isnan, Array(x))
end

function test_logllhood()
    Random.seed!(42)
    dataset = randn(rng, Float32, 32, 32, 1, 500)
    model = init_KAEM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> pu
    model, _, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; MLIR = false, rng = rng)

    x = randn(rng, Float32, 32, 32, 1, b_size) |> pu
    compiled_sample_prior = Reactant.@compile model.sample_prior(model, ps, st_kan, st_lux, st_rng)
    z = first(compiled_sample_prior(model, ps, st_kan, st_lux, st_rng))
    noise = randn(rng, Float32, 32, 32, 1, b_size, b_size) |> pu
    compiled_log_likelihood = Reactant.@compile log_likelihood_IS(z, x, model.lkhood, ps.gen, st_kan.gen, st_lux.gen, noise)
    logllhood, _ = compiled_log_likelihood(z, x, model.lkhood, ps.gen, st_kan.gen, st_lux.gen, noise)
    @test size(logllhood) == (b_size, b_size)
    return @test !any(isnan, Array(logllhood))
end

function test_cnn_generate()
    Random.seed!(42)
    commit!(conf, "CNN", "use_cnn_lkhood", "true")
    dataset = randn(rng, Float32, 32, 32, out_dim, 50)
    model = init_KAEM(dataset, conf, (32, 32, out_dim))
    x_test = first(model.train_loader) |> pu
    model, _, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; MLIR = false, rng = rng)

    compiled_sample_prior = Reactant.@compile model.sample_prior(model, ps, st_kan, st_lux, st_rng)
    z = first(compiled_sample_prior(model, ps, st_kan, st_lux, st_rng))
    compiled_generator = Reactant.@compile model.lkhood.generator(ps.gen, st_kan.gen, st_lux.gen, z)
    x, _ = compiled_generator(ps.gen, st_kan.gen, st_lux.gen, z)
    @test size(x) == (32, 32, out_dim, b_size)
    @test !any(isnan, Array(x))
    return commit!(conf, "CNN", "use_cnn_lkhood", "false")
end

function test_seq_generate()
    Random.seed!(42)
    commit!(conf, "SEQ", "sequence_length", "8")

    dataset = randn(rng, Float32, out_dim, 8, 500)
    model = init_KAEM(dataset, conf, (out_dim, 8))
    x_test = first(model.train_loader) |> pu
    model, _, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; MLIR = false, rng = rng)

    compiled_sample_prior = Reactant.@compile model.sample_prior(model, ps, st_kan, st_lux, st_rng)
    z = first(compiled_sample_prior(model, ps, st_kan, st_lux, st_rng))
    compiled_generator = Reactant.@compile model.lkhood.generator(ps.gen, st_kan.gen, st_lux.gen, z)
    x, _ = compiled_generator(ps.gen, st_kan.gen, st_lux.gen, z)
    @test size(x) == (out_dim, 8, b_size)
    @test !any(isnan, Array(x))
    return commit!(conf, "SEQ", "sequence_length", "1")
end

@testset "KAN Likelihood Tests" begin
    test_generate()
    test_logllhood()
    test_cnn_generate()
    # test_seq_generate()
end
