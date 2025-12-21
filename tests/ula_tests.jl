using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays, Reactant, Statistics

ENV["GPU"] = true

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .KAEM_model

include("../src/KAEM/model_setup.jl")
using .ModelSetup

include("../src/KAEM/ula/unadjusted_langevin.jl")
using .ULA_sampling

include("../src/pipeline/optimizer.jl")
using .optimization

include("../src/KAEM/rng.jl")
using .HLOrng

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
b_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
q_size = last(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))
p_size = first(parse.(Int, retrieve(conf, "EbmModel", "layer_widths")))
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
commit!(conf, "POST_LANGEVIN", "use_langevin", "true")

rng = Random.MersenneTwister(1)
optimizer = create_opt(conf)

function test_basic()
    Random.seed!(42)
    dataset = randn(rng, Float32, 32, 32, 1, 500)
    model = init_KAEM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> pu
    model, _, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; MLIR = false, rng = rng)

    compiled_sampler = Reactant.@compile model.posterior_sampler(ps, st_kan, st_lux, x_test, st_rng)
    z = first(compiled_sampler(ps, st_kan, st_lux, x_test, st_rng))

    P = model.prior.bool_config.mixture_model ? 1 : p_size
    @test size(z) == (q_size, P, b_size, 1)
    @test !any(isnan, Array(z))
    return @test !any(isinf, Array(z))
end

function test_convergence()
    Random.seed!(42)
    dataset = randn(rng, Float32, 32, 32, 1, 500)
    model = init_KAEM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> pu
    model, _, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; MLIR = false, rng = rng)

    sampler = model.posterior_sampler
    compiled_sampler = Reactant.@compile sampler(ps, st_kan, st_lux, x_test, st_rng)

    z1 = first(compiled_sampler(ps, st_kan, st_lux, x_test, st_rng))
    st_rng = seed_rand(model)
    z2 = first(compiled_sampler(ps, st_kan, st_lux, x_test, st_rng))

    # Samples should differ (stochastic)
    z1_arr = Array(z1)
    z2_arr = Array(z2)
    @test any(z1_arr .!= z2_arr)

    sampler_long = initialize_ULA_sampler(model; η = 0.01f0, N = 50)
    compiled_sampler_long = Reactant.@compile sampler_long(ps, st_kan, st_lux, x_test, st_rng)
    z_long = first(compiled_sampler_long(ps, st_kan, st_lux, x_test, st_rng))

    z_long_arr = Array(z_long)
    z_flat = reshape(z_long_arr, :)
    return @test std(z_flat) > 1.0f-1
end

@testset "ULA Convergence Tests" begin
    test_basic()
    test_convergence()
end
