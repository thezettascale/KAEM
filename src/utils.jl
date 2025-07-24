module Utils

export removeNaN, pu, removeZero, removeNeg, half_quant, full_quant, hq, fq, symbol_map

using Lux, Tullio, LinearAlgebra, Statistics, Random, Accessors, BFloat16s
using CUDA, LuxCUDA, KernelAbstractions, Enzyme.EnzymeRules

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

function removeNaN(x)
    return ifelse.(isnan.(x), zero(half_quant), x) |> pu
end

function removeZero(x; ε = half_quant(1e-4))
    return ifelse.(abs.(x) .< ε, ε, x) |> pu
end

function removeNeg(x; ε = half_quant(1e-4))
    return ifelse.(x .< ε, ε, x) |> pu
end

# Num layers must be flexible, yet static, so this is used to index into params/state
const symbol_map = (:a, :b, :c, :d, :e, :f, :g, :h, :i)

EnzymeRules.inactive(::typeof(pu), args...) = nothing
EnzymeRules.inactive(::typeof(half_quant), args...) = nothing
EnzymeRules.inactive(::typeof(full_quant), args...) = nothing

end
