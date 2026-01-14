using Test, Random, LinearAlgebra, Lux, ConfParser, ComponentArrays, Reactant, Optimisers, Statistics
using MLDataDevices: cpu_device

ENV["DEVICE"] = "gpu"

include("../src/baseline/training/trainer.jl")
using .Baseline.VAEModel: VAE, init_VAE, sample
using .Baseline.GANModel: GAN, init_GAN
using .Baseline.DDPMModel: DDPM, init_DDPM
using .Baseline.DDPMSampling: sample_loop, seed_ddpm_rng
using .Baseline.TrainingSetup: prep_vae, prep_gan, prep_ddpm
using .Baseline: Trainer
using .Baseline.Utils: pu

conf = ConfParse("tests/test_baseline_conf.ini")
parse_conf!(conf)

rng = Random.MersenneTwister(42)

include("../src/pipeline/optimizer.jl")
using .optimization: create_opt

optimizer = create_opt(conf)

function test_vae()
    x_shape = (32, 32, 3)
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    model = init_VAE(conf, x_shape; rng = rng)
    x = randn(rng, Float32, x_shape..., batch_size) |> pu

    _, train_step, opt_state, ps, st = prep_vae(model, x, optimizer.rule(); rng = rng)

    ps_before = Array(ps)
    ε = randn(rng, Float32, model.latent_dim, batch_size) |> pu
    loss, ps_new, _, _ = train_step(opt_state, ps, st, x, ε)

    @test !isnan(Float32(loss))
    @test any(Array(ps_new) .!= ps_before)
    return @test !any(isnan, Array(ps_new))
end

function test_vae_sample()
    x_shape = (32, 32, 3)
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    model = init_VAE(conf, x_shape; rng = rng)
    ps = Lux.initialparameters(rng, model) |> ComponentArray |> Lux.f32
    st = Lux.initialstates(rng, model) |> Lux.f32

    z = randn(rng, Float32, model.latent_dim, batch_size)
    x_gen, _ = sample(model, ps, st, z)

    @test size(x_gen) == (x_shape..., batch_size)
    return @test !any(isnan, x_gen)
end

function test_gan()
    x_shape = (32, 32, 3)
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    model = init_GAN(conf, x_shape; rng = rng)
    x_real = randn(rng, Float32, x_shape..., batch_size) |> pu

    _, train_step, opt_state_gen, opt_state_disc, ps, st = prep_gan(
        model, x_real, optimizer.rule(), optimizer.rule(); rng = rng
    )

    ps_before = Array(ps)
    z = randn(rng, Float32, model.latent_dim, batch_size) |> pu
    loss, ps_new, _, _, _ = train_step(
        opt_state_gen, opt_state_disc, ps, st, x_real, z, 1
    )

    ps_new_cpu = Array(ps_new)
    @test !isnan(Float32(loss))
    @test any(ps_new_cpu .!= ps_before)
    return @test !any(isnan, ps_new_cpu)
end

function test_gan_disc()
    x_shape = (32, 32, 3)
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    model = init_GAN(conf, x_shape; rng = rng)
    ps = Lux.initialparameters(rng, model) |> ComponentArray |> Lux.f32
    st = Lux.initialstates(rng, model) |> Lux.f32

    x = randn(rng, Float32, x_shape..., batch_size)
    logits, st_disc = model.discriminator(x, ps.disc, st.disc)

    @test size(logits) == (1, batch_size)
    return @test !any(isnan, logits)
end

function test_ddpm()
    x_shape = (32, 32, 3)
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    model = init_DDPM(conf, x_shape; rng = rng)
    x = randn(rng, Float32, x_shape..., batch_size) |> pu

    _, train_step, opt_state, ps, st = prep_ddpm(model, x, optimizer.rule(); rng = rng)

    ps_before = Array(ps)
    t_idx = rand(rng, 1:model.num_timesteps, batch_size)
    t = Float32.(t_idx) |> pu
    broadcast_shape = (ones(Int, length(x_shape))..., batch_size)
    sqrt_alpha = reshape(model.sqrt_alphas_cumprod_vec[t_idx], broadcast_shape) |> pu
    sqrt_one_minus_alpha = reshape(model.sqrt_one_minus_alphas_cumprod_vec[t_idx], broadcast_shape) |> pu
    noise = randn(rng, Float32, x_shape..., batch_size) |> pu
    loss, ps_new, _, _ = train_step(opt_state, ps, st, x, t, sqrt_alpha, sqrt_one_minus_alpha, noise)

    @test !isnan(Float32(loss))
    @test any(Array(ps_new) .!= ps_before)
    return @test !any(isnan, Array(ps_new))
end

function test_ddpm_q()
    x_shape = (32, 32, 3)
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    model = init_DDPM(conf, x_shape; rng = rng)

    x_0 = randn(rng, Float32, x_shape..., batch_size)
    noise = randn(rng, Float32, x_shape..., batch_size)
    broadcast_shape = (ones(Int, length(x_shape))..., batch_size)

    t_idx = rand(rng, 1:model.num_timesteps, batch_size)
    sqrt_alpha = reshape(model.sqrt_alphas_cumprod_vec[t_idx], broadcast_shape)
    sqrt_one_minus_alpha = reshape(model.sqrt_one_minus_alphas_cumprod_vec[t_idx], broadcast_shape)

    x_noisy = sqrt_alpha .* x_0 .+ sqrt_one_minus_alpha .* noise

    @test size(x_noisy) == size(x_0)
    @test !any(isnan, x_noisy)

    t_early = fill(1, batch_size)
    sqrt_alpha_early = reshape(model.sqrt_alphas_cumprod_vec[t_early], broadcast_shape)
    sqrt_one_minus_alpha_early = reshape(model.sqrt_one_minus_alphas_cumprod_vec[t_early], broadcast_shape)
    x_noisy_early = sqrt_alpha_early .* x_0 .+ sqrt_one_minus_alpha_early .* noise

    t_late = fill(model.num_timesteps, batch_size)
    sqrt_alpha_late = reshape(model.sqrt_alphas_cumprod_vec[t_late], broadcast_shape)
    sqrt_one_minus_alpha_late = reshape(model.sqrt_one_minus_alphas_cumprod_vec[t_late], broadcast_shape)
    x_noisy_late = sqrt_alpha_late .* x_0 .+ sqrt_one_minus_alpha_late .* noise

    signal_early = mean(abs.(x_noisy_early .- noise))
    signal_late = mean(abs.(x_noisy_late .- noise))
    return @test signal_early > signal_late
end

function test_training()
    x_shape = (32, 32, 3)
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    model = init_VAE(conf, x_shape; rng = rng)
    x = randn(rng, Float32, x_shape..., batch_size) |> pu

    _, train_step, opt_state, ps, st = prep_vae(model, x, optimizer.rule(); rng = rng)

    losses = Float32[]
    for i in 1:5
        ε = randn(rng, Float32, model.latent_dim, batch_size) |> pu
        loss, ps, opt_state, st = train_step(opt_state, ps, st, x, ε)
        push!(losses, Float32(loss))
    end

    @test !any(isnan, losses)
    @test all(losses .< 1.0e6)
    return @test losses[end] < losses[1] * 2
end

function test_sample_loop()
    x_shape = (32, 32, 3)
    batch_size = 10
    stride = 10

    model = init_DDPM(conf, x_shape; rng = rng)
    x = randn(rng, Float32, x_shape..., batch_size) |> pu
    _, _, _, ps, st = prep_ddpm(model, x, optimizer.rule(); rng = rng)

    st_rng = seed_ddpm_rng(model, x_shape, batch_size, stride; rng = rng)
    loop_compiled = Reactant.@compile sample_loop(
        model.unet, ps, Lux.testmode(st), st_rng, batch_size
    )

    st_rng_new = seed_ddpm_rng(model, x_shape, batch_size, stride; rng = rng)
    x_gen, st_final = loop_compiled(
        model.unet, ps, Lux.testmode(st), st_rng_new, batch_size
    )

    x_gen_cpu = Array(x_gen)
    @test size(x_gen_cpu) == (x_shape..., batch_size)
    @test !any(isnan, x_gen_cpu)
    @test all(x_gen_cpu .>= 0.0f0)
    return @test all(x_gen_cpu .<= 1.0f0)
end

function test_vae_gen()
    trainer = Baseline.init_trainer(:vae, conf, "CIFAR10"; rng = rng, MLIR = true)
    x_gen, _ = trainer.generate_batch_fn(
        trainer.gen_compiled, trainer.ps, trainer.st,
        trainer.rng, trainer.x_shape, trainer.batch_size
    )
    x_gen_cpu = Array(x_gen)
    @test size(x_gen_cpu) == (trainer.x_shape..., trainer.batch_size)
    @test !any(isnan, x_gen_cpu)
    @test all(x_gen_cpu .>= 0.0f0)
    @test all(x_gen_cpu .<= 1.0f0)
    return @test true
end

function test_ddpm_gen()
    trainer = Baseline.init_trainer(:ddpm, conf, "CIFAR10"; rng = rng, MLIR = true)
    x_gen, _ = trainer.generate_batch_fn(
        trainer.gen_compiled, trainer.ps, trainer.st,
        trainer.rng, trainer.x_shape, trainer.batch_size
    )
    x_gen_cpu = Array(x_gen)
    @test size(x_gen_cpu) == (trainer.x_shape..., trainer.batch_size)
    @test !any(isnan, x_gen_cpu)
    @test all(x_gen_cpu .>= 0.0f0)
    @test all(x_gen_cpu .<= 1.0f0)
    return @test true
end

function test_gan_gen()
    trainer = Baseline.init_trainer(:gan, conf, "CIFAR10"; rng = rng, MLIR = true)
    x_gen, _ = trainer.generate_batch_fn(
        trainer.gen_compiled, trainer.ps, trainer.st,
        trainer.rng, trainer.x_shape, trainer.batch_size
    )
    x_gen_cpu = Array(x_gen)
    @test size(x_gen_cpu) == (trainer.x_shape..., trainer.batch_size)
    @test !any(isnan, x_gen_cpu)
    @test all(x_gen_cpu .>= 0.0f0)
    @test all(x_gen_cpu .<= 1.0f0)
    return @test true
end

@testset "Baseline Tests" begin
    @testset "VAE" begin
        test_vae()
        test_vae_sample()
    end

    @testset "GAN" begin
        test_gan()
        test_gan_disc()
    end

    @testset "DDPM" begin
        test_ddpm()
        test_ddpm_q()
    end

    @testset "Training" begin
        test_training()
    end
end

@testset "Single-function Tests" begin
    @testset "sample_loop compiled" begin
        test_sample_loop()
    end

    @testset "Gen compile" begin
        test_vae_gen()
        test_ddpm_gen()
        test_gan_gen()
    end
end
