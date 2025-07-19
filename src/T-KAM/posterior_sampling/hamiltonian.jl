module HamiltonianDynamics

export position_update, momentum_update, leapfrop_proposal

using CUDA, KernelAbstractions, Tullio, Lux, LuxCUDA, ComponentArrays

include("../../utils.jl")
using .Utils: full_quant, half_quant

function position_update(
    z::AbstractArray{U},
    momentum::AbstractArray{U},
    ∇z::AbstractArray{U},
    M::AbstractArray{U},
    η::AbstractArray{U},
)::Tuple{AbstractArray{U},AbstractArray{U}} where {U<:full_quant}
    η = reshape(η, 1, 1, size(η)...)
    y = momentum .+ (η ./ 2) .* ∇z ./ M
    ẑ = z .+ η .* y ./ M
    return y, ẑ
end

function momentum_update(
    y::AbstractArray{U},
    ∇ẑ::AbstractArray{U},
    M::AbstractArray{U},
    η::AbstractArray{U},
)::AbstractArray{U} where {U<:full_quant}
    return y .+ (reshape(η, 1, 1, size(η)...) ./ 2) .* ∇ẑ ./ M
end

function leapfrop_proposal(
    z::AbstractArray{U},
    ∇z::AbstractArray{U},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    logpos_z::AbstractArray{U},
    momentum::AbstractArray{U},  # This is y = M^{-1/2}p
    M::AbstractArray{U},         # This is M^{1/2}
    η::AbstractArray{U},
    logpos_withgrad::Function,
    model,
    ps::ComponentArray{T},
    st::NamedTuple,
)::Tuple{
    AbstractArray{U},
    AbstractArray{U},
    AbstractArray{U},
    AbstractArray{U},
    AbstractArray{U},
    NamedTuple,
} where {T<:half_quant,U<:full_quant}
    """
    Implements preconditioned Hamiltonian dynamics with transformed momentum:
    y*(x,y)   = y  + (eps/2)M^{-1/2}grad(log pi)(x)
    x'(x,y*)  = x  + eps M^{-1/2}y*
    y'(x',y*) = y* + (eps/2)M^{-1/2}grad(log pi)(x')
    """
    # # Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad) and full step position update
    p, ẑ = position_update(z, momentum, ∇z, M, η)

    # Get gradient at new position
    logpos_ẑ, ∇ẑ, st = logpos_withgrad(ẑ, ∇z, x, temps, model, ps, st)

    # Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad)
    p = momentum_update(p, ∇ẑ, M, η)

    # Hamiltonian difference for transformed momentum
    # H(x,y) = -log(pi(x)) + (1/2)||p||^2 since p ~ N(0,I)
    log_r =
        logpos_ẑ - logpos_z -
        dropdims(
            sum(p .^ 2; dims = (1, 2)) - sum(momentum .^ 2; dims = (1, 2));
            dims = (1, 2),
        ) ./ 2

    return ẑ, logpos_ẑ, ∇ẑ, -p, log_r, st
end

end
