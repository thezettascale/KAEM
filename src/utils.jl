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
    # @tullio NaNs[i, j, k] := isnan(x[i, j, k])
    # x = ifelse.(NaNs, 0f0, x)
    # return device(x)
    return CUDA.@allowscalar replace(x, NaN => 0f0)
end

function removeZero(x; ε=1f-3)
    @tullio zeros[i, j, k] := abs(x[i, j, k]) .< ε
    x = ifelse.(zeros, ε, x)
    return device(x)
end

function next_rng(seed)
    rng = @ignore_derivatives Random.seed!(seed)
    return seed + 1, rng
end

end