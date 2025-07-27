module Utils

export pu, half_quant, full_quant, hq, fq, symbol_map, activation_mapping

using Lux, LinearAlgebra, Statistics, Random, Accessors, BFloat16s, CUDA, KernelAbstractions, LuxCUDA, Enzyme.EnzymeRules, NNlib, Reactant, MLDataDevices

if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

const pu = MLDataDevices.reactant_device()

# # Mixed precision - sometimes unstable, use FP16 when Tensor Cores are available
const QUANT_MAP =
    Dict("BF16" => BFloat16, "FP16" => Float16, "FP32" => Float32, "FP64" => Float64)

const LUX_QUANT_MAP =
    Dict("BF16" => Lux.bf16, "FP16" => Lux.f16, "FP32" => Lux.f32, "FP64" => Lux.f64)

const half_quant = get(QUANT_MAP, uppercase(get(ENV, "HALF_QUANT", "FP32")), Float32)
const full_quant = get(QUANT_MAP, uppercase(get(ENV, "FULL_QUANT", "FP32")), Float32)
const hq = get(LUX_QUANT_MAP, uppercase(get(ENV, "HALF_QUANT", "FP32")), Lux.f32)
const fq = get(LUX_QUANT_MAP, uppercase(get(ENV, "FULL_QUANT", "FP32")), Lux.f32)

# Num layers must be flexible, yet static, so this is used to index into params/state
const symbol_map = (:a, :b, :c, :d, :e, :f, :g, :h, :i)

const activation_mapping = Dict(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.tanh_fast,
    "sigmoid" => NNlib.sigmoid_fast,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu,
    "selu" => NNlib.selu,
    "tanh" => NNlib.tanh_fast,
    "silu" => x -> x .* NNlib.sigmoid_fast(x),
    "elu" => NNlib.elu,
    "celu" => NNlib.celu,
    "none" => x -> x .* zero(half_quant),
)

end
