module Utils

export removeNaN, device, removeZero, next_rng, quant, resample_idx

using Lux, Tullio, LinearAlgebra, Statistics, Random
using CUDA, LuxCUDA, KernelAbstractions
using ChainRules: @ignore_derivatives

const pu = CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false")) ? gpu_device() : cpu_device()

# Only FP32 is supported for now
const quant = Dict(
    "FP32" => Float32,
)[get(ENV, "QUANT", "FP32")]

function device(x)
    return pu(x)
end

function removeNaN(x)
    return ifelse.(isnan.(x), quant(0), x) |> device
end

function removeZero(x; ε=quant(1e-4))
    return ifelse.(abs.(x) .< ε, ε, x) |> device
end

function next_rng(seed)
    rng = @ignore_derivatives Random.seed!(seed)
    return seed + 1, rng
end

function find_resampled_indices!(resampled_indices, cdf, u, B, N)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= B * N
        b = div(i - 1, N) + 1  
        j = mod(i - 1, N) + 1  

        for k in 1:N
            if cdf[b, k] >= u[b, j]
                resampled_indices[b, j] = k
                break
            end
        end
    end
    return
end

function cuda_resampling(cdf::CuArray, u::CuArray, B::Int, N::Int)
    """Find the index of the first cdf value greater than the random value"""
    resampled_indices = CUDA.zeros(Int, B, N)
    threads_per_block = 256
    num_blocks = cld(B * N, threads_per_block)
    @cuda threads=threads_per_block blocks=num_blocks find_resampled_indices!(resampled_indices, cdf, u, B, N)
    resampled_indices = resampled_indices |> cpu_device()
    replace!(resampled_indices, 0=>size(cdf,2))
    return resampled_indices
end

function cpu_resampling(cdf::AbstractArray, u::AbstractArray, B::Int, N::Int)
    resampled_indices = reduce(hcat, map(b -> searchsortedfirst.(Ref(cdf[b, :]), u[b, :]), 1:B))
    replace!(resampled_indices, size(cdf,2)+1 => size(cdf,2))
    return resampled_indices
end

const resample_idx = CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false")) ? cuda_resampling : cpu_resampling

end