module Utils

export removeNaN, device, removeZero, removeNeg, half_quant, full_quant, hq, fq, symbol_map

using Lux, Tullio, LinearAlgebra, Statistics, Random, Accessors, BFloat16s, Reactant, MLDataDevices
using CUDA, LuxCUDA, KernelAbstractions, Enzyme.EnzymeRules

if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

const pu = MLDataDevices.reactant_device()

# Mixed precision - sometimes unstable, use FP16 when Tensor Cores are available
const QUANT_MAP =
    Dict("BF16" => BFloat16, "FP16" => Float16, "FP32" => Float32, "FP64" => Float64)

const LUX_QUANT_MAP =
    Dict("BF16" => Lux.bf16, "FP16" => Lux.f16, "FP32" => Lux.f32, "FP64" => Lux.f64)

const half_quant = get(QUANT_MAP, uppercase(get(ENV, "HALF_QUANT", "FP32")), Float32)
const full_quant = get(QUANT_MAP, uppercase(get(ENV, "FULL_QUANT", "FP32")), Float32)
const hq = get(LUX_QUANT_MAP, uppercase(get(ENV, "HALF_QUANT", "FP32")), Lux.f32)
const fq = get(LUX_QUANT_MAP, uppercase(get(ENV, "FULL_QUANT", "FP32")), Lux.f32)

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

# Num layers must be flexible, yet static, so this is used to index into params/state
const symbol_map = (:a, :b, :c, :d, :e, :f, :g, :h, :i)

EnzymeRules.inactive(::typeof(device), args...) = nothing
EnzymeRules.inactive(::typeof(half_quant), args...) = nothing
EnzymeRules.inactive(::typeof(full_quant), args...) = nothing

end
