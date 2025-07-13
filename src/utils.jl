module Utils

export removeNaN, device, removeZero, removeNeg, next_rng, half_quant, full_quant, hq, fq

using Lux, Tullio, LinearAlgebra, Statistics, Random, Accessors, BFloat16s
using CUDA, LuxCUDA, KernelAbstractions
using ChainRules: @ignore_derivatives

const pu =
    CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false")) ? gpu_device() : cpu_device()

# Mixed precision - sometimes unstable, use FP16 when Tensor Cores are available
const half_quant = Dict("BF16" => BFloat16, "FP16" => Float16, "FP32" => Float32)[get(
    ENV,
    "HALF_QUANT",
    "FP32",
)]

const full_quant = Dict("FP16" => Float16, "FP32" => Float32, "FP64" => Float64)[get(
    ENV,
    "FULL_QUANT",
    "FP32",
)]

const hq = Dict("BF16" => Lux.bf16, "FP16" => Lux.f16, "FP32" => Lux.f32)[get(
    ENV,
    "HALF_QUANT",
    "FP32",
)]

const fq = Dict("FP16" => Lux.f16, "FP32" => Lux.f32, "FP64" => Lux.f64)[get(
    ENV,
    "FULL_QUANT",
    "FP32",
)]

function device(x)
    return pu(x)
end

function removeNaN(x)
    return ifelse.(isnan.(x), zero(half_quant), x) |> device
end

function removeZero(x; ε = half_quant(1e-4))
    return ifelse.(abs.(x) .< ε, ε, x) |> device
end

function removeNeg(x; ε = half_quant(1e-4))
    return ifelse.(x .< ε, ε, x) |> device
end

function next_rng(seed)
    rng = @ignore_derivatives Random.seed!(seed)
    return seed + 1, rng
end

end
