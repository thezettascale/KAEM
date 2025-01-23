module Utils

export removeNaN, device, removeZero, removeNeg, next_rng, quant

using Lux, Tullio, LinearAlgebra, Statistics, Random, Accessors
using CUDA, LuxCUDA, KernelAbstractions
using ChainRules: @ignore_derivatives

const pu = CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false")) ? gpu_device() : cpu_device()

# Only FP32 is supported for now
const quant = Dict(
    "FP32" => Float32,
    "FP64" => Float64
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

function removeNeg(x; ε=quant(1e-4))
    return ifelse.(x .< ε, ε, x) |> device
end

function next_rng(seed)
    rng = @ignore_derivatives Random.seed!(seed)
    return seed + 1, rng
end

function move_to_cpu(model, ps, st)
    ps, st = ps |> cpu_device(), st |> cpu_device()
    
    for i in 1:model.prior.depth
        @reset model.prior.fcns_qp[Symbol("$i")].grid = model.prior.fcns_qp[Symbol("$i")].grid |> cpu_device()
    end

    for i in 1:model.lkhood.depth
        @reset model.lkhood.Φ_fcns[Symbol("$i")].grid = model.lkhood.Φ_fcns[Symbol("$i")].grid |> cpu_device()
    end

    return model, ps, st
end

function move_to_gpu(model, ps, st)
    ps, st = ps |> deivce, st |> device

    for i in 1:model.prior.depth
        @reset model.prior.fcns_qp[Symbol("$i")].grid = model.prior.fcns_qp[Symbol("$i")].grid |> device
    end

    for i in 1:model.lkhood.depth
        @reset model.lkhood.Φ_fcns[Symbol("$i")].grid = model.lkhood.Φ_fcns[Symbol("$i")].grid |> device
    end

    return model, ps, st
end

end