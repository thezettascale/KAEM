module Utils

export removeNaN, device, removeZero, next_rng

using Lux, Tullio, LinearAlgebra, Statistics, Random
using CUDA, LuxCUDA, KernelAbstractions
using ChainRules: @ignore_derivatives

const pu = CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false")) ? gpu_device() : cpu_device()

function device(x)
    return pu(x)
end

function removeNaN(x)
    return ifelse.(isnan.(x), 0f0, x) |> device
end

function removeZero(x; ε=1f-4)
    return ifelse.(abs.(x) .< ε, ε, x) |> device
end

function next_rng(seed)
    rng = @ignore_derivatives Random.seed!(seed)
    return seed + 1, rng
end

end