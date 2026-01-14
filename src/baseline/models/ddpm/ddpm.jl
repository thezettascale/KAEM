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
    sampling_stride::Int
    sampling_num_steps::Int
    sampling_timesteps::AbstractArray{T}
    sampling_alphas::AbstractArray{T}
    sampling_alphas_cumprod::AbstractArray{T}
    sampling_betas::AbstractArray{T}
    sampling_noise_masks::AbstractArray{T}
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
    sampling_stride = parse(Int, retrieve(conf, "DDPM", "sampling_stride"))

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

    # Strided denoising schedule
    sampling_num_steps = cld(num_timesteps, sampling_stride)
    timesteps_idx = [max(num_timesteps - (i - 1) * sampling_stride, 1) for i in 1:sampling_num_steps]

    ndims_x = length(x_shape) + 1  # (H, W, C, batch)
    broadcast_shape = (ones(Int, ndims_x)..., sampling_num_steps)

    sampling_timesteps = repeat(Float32.(timesteps_idx)', batch_size, 1)
    sampling_alphas = reshape(alphas[timesteps_idx], broadcast_shape...)
    sampling_alphas_cumprod = reshape(alphas_cumprod[timesteps_idx], broadcast_shape...)
    sampling_betas = reshape(betas[timesteps_idx], broadcast_shape...)
    sampling_noise_masks = reshape(vcat(ones(Float32, sampling_num_steps - 1), 0.0f0), broadcast_shape...)

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
        sampling_stride,
        sampling_num_steps,
        sampling_timesteps,
        sampling_alphas,
        sampling_alphas_cumprod,
        sampling_betas,
        sampling_noise_masks,
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
