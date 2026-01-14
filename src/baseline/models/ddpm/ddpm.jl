module DDPMModel

export DDPM, init_DDPM

using Lux, ConfParser, Random

using ..Utils

include("unet.jl")
using .UNetArchitecture

struct DDPM{T <: Float32} <: Lux.AbstractLuxLayer
    unet::UNet
    num_timesteps::Int
    beta_start::T
    beta_end::T
    betas::AbstractVector{T}
    alphas::AbstractVector{T}
    alphas_cumprod::AbstractVector{T}
    sqrt_alphas_cumprod::AbstractArray{T}
    sqrt_one_minus_alphas_cumprod::AbstractArray{T}
    sqrt_alphas_cumprod_vec::AbstractVector{T}
    sqrt_one_minus_alphas_cumprod_vec::AbstractVector{T}
    x_shape::Tuple{Vararg{Int}}
    batch_size::Int
end

function init_DDPM(
        conf::ConfParse,
        x_shape::Tuple{Vararg{Int}};
        rng::AbstractRNG = Random.default_rng(),
    )
    num_timesteps = parse(Int, retrieve(conf, "DDPM", "num_timesteps"))
    beta_start = parse(Float32, retrieve(conf, "DDPM", "beta_start"))
    beta_end = parse(Float32, retrieve(conf, "DDPM", "beta_end"))
    channels = parse.(Int, retrieve(conf, "DDPM", "channels"))
    kernel_size = parse(Int, retrieve(conf, "DDPM", "kernel_size"))
    time_dim = parse(Int, retrieve(conf, "DDPM", "time_dim"))
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    in_channels = last(x_shape)

    # Linear beta schedule
    betas = collect(range(beta_start, beta_end, length = num_timesteps))
    alphas = 1.0f0 .- betas
    alphas_cumprod = cumprod(alphas)
    sqrt_alphas_cumprod_vec = sqrt.(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod_vec = sqrt.(1.0f0 .- alphas_cumprod)
    sqrt_alphas_cumprod = reshape(
        sqrt_alphas_cumprod_vec,
        ones(Int, length(x_shape))...,
        num_timesteps
    )
    sqrt_one_minus_alphas_cumprod = reshape(
        sqrt_one_minus_alphas_cumprod_vec,
        ones(Int, length(x_shape))...,
        num_timesteps
    )

    unet = init_unet(in_channels, channels, kernel_size, time_dim)

    return DDPM{Float32}(
        unet,
        num_timesteps,
        beta_start,
        beta_end,
        betas,
        alphas,
        alphas_cumprod,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        sqrt_alphas_cumprod_vec,
        sqrt_one_minus_alphas_cumprod_vec,
        x_shape,
        batch_size,
    )
end

function Lux.initialparameters(rng::AbstractRNG, model::DDPM)
    return Lux.initialparameters(rng, model.unet)
end

function Lux.initialstates(rng::AbstractRNG, model::DDPM)
    return Lux.initialstates(rng, model.unet)
end

function (model::DDPM)(x_noisy, t, ps, st)
    return model.unet(x_noisy, t, ps, st)
end

end
