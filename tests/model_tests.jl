using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays, Reactant
using MultivariateStats: reconstruct
using MLDataDevices: cpu_device

ENV["GPU"] = "gpu"

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .KAEM_model

include("../src/KAEM/model_setup.jl")
using .ModelSetup

include("../src/KAEM/grid_updating.jl")
using .ModelGridUpdating

include("../src/pipeline/optimizer.jl")
using .optimization

conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)
out_dim = parse(Int, retrieve(conf, "GeneratorModel", "output_dim"))
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
commit!(conf, "VARIATIONAL", "use_variational", "false")

optimizer = create_opt(conf)
rng = Random.MersenneTwister(1)

function test_ps_derivative()
    dataset = randn(rng, Float32, 32, 32, 1, 500)
    model = init_KAEM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> pu
    model, opt_state, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; rng = rng)

    ps_before = Array(ps)
    loss, ps, _, st_ebm, st_gen =
        model.train_step(opt_state, ps, st_kan, st_lux, x_test, 1, st_rng)

    ps_after = Array(ps)
    @test any(ps_before .!= ps_after)
    return @test !any(isnan, ps_after)
end

function test_grid_update()
    dataset = randn(rng, Float32, 32, 32, 1, 500)
    model = init_KAEM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> pu
    model, opt_state, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; rng = rng)

    x = first(model.train_loader) |> pu

    updater = GridUpdater(model, conf)
    compiled_update = Reactant.@compile updater(
        x,
        ps,
        st_kan,
        Lux.testmode(st_lux),
        1,
        st_rng,
    )

    before = st_kan |> cpu_device() |> ComponentArray
    ps_before = Array(ps)

    ps, st_kan, st_lux = compiled_update(
        x,
        ps,
        st_kan,
        Lux.testmode(st_lux),
        1,
        st_rng
    )

    grid = st_kan |> cpu_device() |> ComponentArray
    @test !all(iszero, grid - before)
    @test !any(isnan, grid)
    @test !all(iszero, Array(ps) - ps_before)
    @test !any(isnan, Array(ps))

    ps_before = Array(ps)
    loss, ps, _, st_ebm, st_gen =
        model.train_step(opt_state, ps, st_kan, st_lux, x_test, 1, st_rng)

    ps_after = Array(ps)
    @test any(ps_before .!= ps_after)
    return @test !any(isnan, ps_after)
end

function test_pca()
    dataset = randn(rng, Float32, 32, 32, 1, 500)
    commit!(conf, "PCA", "use_pca", "true")
    commit!(conf, "PCA", "pca_components", "10")
    model = init_KAEM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader)
    model, opt_state, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; rng = rng)

    @test size(x_test, 1) == 10

    x_recon = reconstruct(model.PCA_model, cpu_device()(x_test))
    x_recon = reshape(x_recon, model.original_data_size..., :)
    commit!(conf, "PCA", "use_pca", "false")
    return @test all(size(x_recon)[1:3] .== size(dataset)[1:3])
end

function test_mala_loss()
    dataset = randn(rng, Float32, 32, 32, 1, 500)
    commit!(conf, "POST_LANGEVIN", "use_langevin", "true")
    model = init_KAEM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> pu
    model, opt_state, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; rng = rng)

    ps_before = Array(ps)
    loss, ps, _, st_ebm, st_gen =
        model.train_step(opt_state, ps, st_kan, st_lux, x_test, 1, st_rng)

    ps_after = Array(ps)
    @test any(ps_before .!= ps_after)
    return @test !any(isnan, ps_after)
end

function test_cnn_loss()
    dataset = randn(rng, Float32, 32, 32, 3, 500)
    commit!(conf, "CNN", "use_cnn_lkhood", "true")
    commit!(conf, "CNN", "latent_concat", "false")
    commit!(conf, "PCA", "use_pca", "false")
    model = init_KAEM(dataset, conf, (32, 32, 3))
    x_test = first(model.train_loader) |> pu
    model, opt_state, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; rng = rng)

    ps_before = Array(ps)
    loss, ps, _, st_ebm, st_gen =
        model.train_step(opt_state, ps, st_kan, st_lux, x_test, 1, st_rng)

    ps_after = Array(ps)
    @test any(ps_before .!= ps_after)
    @test !any(isnan, ps_after)
    return commit!(conf, "CNN", "use_cnn_lkhood", "false")
end

function test_cnn_residual_loss()
    dataset = randn(rng, Float32, 32, 32, 3, 500)
    commit!(conf, "CNN", "use_cnn_lkhood", "true")
    commit!(conf, "CNN", "latent_concat", "true")
    model = init_KAEM(dataset, conf, (32, 32, 3))
    x_test = first(model.train_loader) |> pu
    model, opt_state, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; rng = rng)

    ps_before = Array(ps)
    loss, ps, _, st_ebm, st_gen =
        model.train_step(opt_state, ps, st_kan, st_lux, x_test, 1, st_rng)

    ps_after = Array(ps)
    @test any(ps_before .!= ps_after)
    commit!(conf, "CNN", "use_cnn_lkhood", "false")
    return @test !any(isnan, ps_after)
end

function test_seq_loss()
    dataset = randn(rng, Float32, 50, 10, 500)
    commit!(conf, "SEQ", "sequence_length", "10")
    commit!(conf, "SEQ", "vocab_size", "50")
    model = init_KAEM(dataset, conf, (50, 10))
    x_test = first(model.train_loader) |> pu
    model, opt_state, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; rng = rng)

    ps_before = Array(ps)
    loss, ps, _, st_ebm, st_gen =
        model.train_step(opt_state, ps, st_kan, st_lux, x_test, 1, st_rng)

    ps_after = Array(ps)
    @test any(ps_before .!= ps_after)
    return @test !any(isnan, ps_after)
end

@testset "KAEM Tests" begin
    test_ps_derivative()
    test_grid_update()
    test_pca()
    test_mala_loss()
    test_cnn_loss()
    test_cnn_residual_loss()
    # test_seq_loss()
end
