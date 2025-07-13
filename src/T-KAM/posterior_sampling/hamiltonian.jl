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
    nd = ndims(z)
    η_half = η ./ 2

    if nd == 4
        y = momentum .+ (η_half[reshape(1,1,:,:)] .* ∇z) ./ M
        ẑ = z .+ (η[reshape(1,1,:,:)] .* y) ./ M
    elseif nd == 3
        y = momentum .+ (η_half[reshape(1,1,:)] .* ∇z) ./ M
        ẑ = z .+ (η[reshape(1,1,:)] .* y) ./ M
    else
        throw(ArgumentError("position_update only supports 3D or 4D arrays"))
    end

    return y, ẑ
end

function momentum_update(
    y::AbstractArray{U},
    ∇ẑ::AbstractArray{U},
    M::AbstractArray{U},
    η::AbstractArray{U},
    ) where {U<:full_quant}
    nd = ndims(y)
    η_half = η ./ 2

    if nd == 4
        y_out = y .+ (η_half[reshape(1,1,:,:)] .* ∇ẑ) ./ M
    elseif nd == 3
        y_out = y .+ (η_half[reshape(1,1,:)] .* ∇ẑ) ./ M
    else
        throw(ArgumentError("momentum_update only supports 3D or 4D arrays"))
    end
    
    return y_out
end

end