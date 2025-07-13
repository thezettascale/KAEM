module HamiltonianDynamics

export position_update, momentum_update

using CUDA, KernelAbstractions, Tullio, Lux, LuxCUDA

include("../../utils.jl")
using .Utils: full_quant

function position_update(
    z::AbstractArray{U},
    momentum::AbstractArray{U},
    ∇z::AbstractArray{U},
    M::AbstractArray{U},
    η::AbstractArray{U},
) where {U<:full_quant}
    η = reshape(η, 1, 1, size(η)...)
    y = momentum .+ (η ./ 2) .* ∇z ./ M
    ẑ = z .+ η .* y ./ M
    return y, ẑ
end

function momentum_update(
    y::AbstractArray{U},
    ∇ẑ::AbstractArray{U},
    M::AbstractArray{U},
    η::AbstractArray{U}
) where {U<:full_quant}
    return y .+ (reshape(η, 1, 1, size(η)...) ./ 2) .* ∇ẑ ./ M
end

end 
