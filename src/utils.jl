module Utils

export removeNaN, device, removeZero, removeNeg, next_rng, half_quant, full_quant, move_to_cpu, move_to_gpu, hq, fq

using Lux, Tullio, LinearAlgebra, Statistics, Random, Accessors
using CUDA, LuxCUDA, KernelAbstractions
using ChainRules: @ignore_derivatives

const pu = CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false")) ? gpu_device() : cpu_device()

# Mixed precision - sometimes unstable, use FP16 when Tensor Cores are available
const half_quant = Dict(
    "FP16" => Float16,
    "FP32" => Float32
)[get(ENV, "HALF_QUANT", "FP32")]

const full_quant = Dict(
    "FP16" => Float16,
    "FP32" => Float32,
    "FP64" => Float64,
)[get(ENV, "FULL_QUANT", "FP32")]

const hq = half_quant == Float16 ? Lux.f16 : Lux.f32
const fq = full_quant == Float16 ? Lux.f16 : (full_quant == Float64 ? Lux.f64 : Lux.f32)

function device(x)
    return pu(x)
end

function removeNaN(x)
    return ifelse.(isnan.(x), half_quant(0), x) |> device
end

function removeZero(x; ε=half_quant(1e-4))
    return ifelse.(abs.(x) .< ε, ε, x) |> device
end

function removeNeg(x; ε=half_quant(1e-4))
    return ifelse.(x .< ε, ε, x) |> device
end

function next_rng(seed)
    rng = @ignore_derivatives Random.seed!(seed)
    return seed + 1, rng
end

function move_to_cpu(model, ps, st)
    ps, st = ps |> cpu_device() |> Lux.f32, st |> cpu_device() |> Lux.f32
    
    for i in 1:model.prior.depth
        @reset model.prior.fcns_qp[Symbol("$i")].grid = model.prior.fcns_qp[Symbol("$i")].grid |> cpu_device() |> Lux.f32
    end

    if !model.lkhood.CNN
        for i in 1:model.lkhood.depth
            @reset model.lkhood.Φ_fcns[Symbol("$i")].grid = model.lkhood.Φ_fcns[Symbol("$i")].grid |> cpu_device() |> Lux.f32
        end
    end

    return model, ps, st
end

function move_to_gpu(model, ps, st)
    ps, st = ps |> deivce, st |> device

    for i in 1:model.prior.depth
        @reset model.prior.fcns_qp[Symbol("$i")].grid = model.prior.fcns_qp[Symbol("$i")].grid |> device
    end

    if !model.lkhood.CNN
        for i in 1:model.lkhood.depth
            @reset model.lkhood.Φ_fcns[Symbol("$i")].grid = model.lkhood.Φ_fcns[Symbol("$i")].grid |> device
        end
    end

    return model, ps, st
end

end