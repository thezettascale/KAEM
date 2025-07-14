module Utils

export removeNaN,
    device, removeZero, removeNeg, next_rng, half_quant, full_quant, hq, fq, AD_backend

using Lux, Tullio, LinearAlgebra, Statistics, Random, Accessors, BFloat16s
using CUDA, LuxCUDA, KernelAbstractions, Zygote, Enzyme, Enzyme.EnzymeRules
using ChainRules: @ignore_derivatives

const pu =
    CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false")) ? gpu_device() : cpu_device()

# # Mixed precision - sometimes unstable, use FP16 when Tensor Cores are available
const QUANT_MAP =
    Dict("BF16" => BFloat16, "FP16" => Float16, "FP32" => Float32, "FP64" => Float64)

const LUX_QUANT_MAP =
    Dict("BF16" => Lux.bf16, "FP16" => Lux.f16, "FP32" => Lux.f32, "FP64" => Lux.f64)

const half_quant = get(QUANT_MAP, uppercase(get(ENV, "HALF_QUANT", "FP32")), Float32)
const full_quant = get(QUANT_MAP, uppercase(get(ENV, "FULL_QUANT", "FP32")), Float32)
const hq = get(LUX_QUANT_MAP, uppercase(get(ENV, "HALF_QUANT", "FP32")), Lux.f32)
const fq = get(LUX_QUANT_MAP, uppercase(get(ENV, "FULL_QUANT", "FP32")), Lux.f32)

# Automatic differentiation
const AD_BACKEND_MAP = Dict(
    "ZYGOTE" => AutoZygote(),
    "ENZYME" => AutoEnzyme(;
        function_annotation = Enzyme.Duplicated,
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    ),
)

const AD_backend =
    get(AD_BACKEND_MAP, uppercase(get(ENV, "AD_BACKEND", "ZYGOTE")), AutoZygote())

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

EnzymeRules.inactive(::typeof(device), args...) = nothing
EnzymeRules.inactive(::typeof(next_rng), args...) = nothing

end
