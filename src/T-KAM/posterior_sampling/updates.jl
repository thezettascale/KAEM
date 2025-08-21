module LangevinUpdates

export update_z!, logpos_withgrad, leapfrog

using CUDA, KernelAbstractions, Lux, LuxCUDA, ComponentArrays, Accessors, Tullio

using ..Utils
using ..T_KAM_model

include("log_posteriors.jl")
using .LogPosteriors: autoMALA_value_and_grad

## ULA ##
function update_z!(
    z::AbstractArray{U,3},
    ∇z::AbstractArray{U,3},
    η::U,
    ξ::AbstractArray{U,3},
    sqrt_2η::U,
    Q::Int,
    P::Int,
    S::Int,
)::Nothing where {U<:full_quant}
    @tullio z[q, p, s] = z[q, p, s] + η * ∇z[q, p, s] + sqrt_2η * ξ[q, p, s]
    return nothing
end

## autoMALA ##
function position_update!(
    z::AbstractArray{U,3},
    momentum::AbstractArray{U,3}, # p*
    ∇z::AbstractArray{U,3},
    M::AbstractArray{U,2},
    η::AbstractArray{U,1},
)::Nothing where {U<:full_quant}
    @tullio momentum[q, p, s] = momentum[q, p, s] + (η[s] / 2) * ∇z[q, p, s] / M[q, p]
    @tullio z[q, p, s] = z[q, p, s] + η[s] * momentum[q, p, s] / M[q, p]
    return nothing
end

function momentum_update!(
    momentum::AbstractArray{U,3}, # p*
    ∇ẑ::AbstractArray{U,3},
    M::AbstractArray{U,2},
    η::AbstractArray{U,1},
)::Nothing where {U<:full_quant}
    @tullio momentum[q, p, s] = momentum[q, p, s] + (η[s] / 2) * ∇ẑ[q, p, s] / M[q, p]
    return nothing
end

function logpos_withgrad(
    z::AbstractArray{T,3},
    ∇z::AbstractArray{T,3},
    x::AbstractArray{T},
    temps::AbstractArray{T,1},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::Tuple{
    AbstractArray{U,1},
    AbstractArray{U,3},
    NamedTuple,
} where {T<:half_quant,U<:full_quant}
    logpos, ∇z_k, st_ebm, st_gen =
        autoMALA_value_and_grad(z, ∇z, x, temps, model, ps, st_kan, st_lux)
    @reset st_lux.ebm = st_ebm
    @reset st_lux.gen = st_gen

    return U.(logpos) ./ model.loss_scaling.full,
    U.(∇z_k) ./ model.loss_scaling.full,
    st_lux
end

function leapfrog(
    z::AbstractArray{U,3},
    ∇z::AbstractArray{U,3},
    x::AbstractArray{T},
    temps::AbstractArray{T,1},
    logpos_z::AbstractArray{U,1},
    p::AbstractArray{U,3},  # This is momentum = M^{-1/2}p
    M::AbstractArray{U,2},         # This is M^{1/2}
    η::AbstractArray{U,1},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::Tuple{
    AbstractArray{U,3},
    AbstractArray{U,1},
    AbstractArray{U,3},
    AbstractArray{U,3},
    AbstractArray{U,1},
    NamedTuple,
} where {T<:half_quant,U<:full_quant}
    """
    Implements preconditioned Hamiltonian dynamics with transformed momentum:
    y*(x,y)   = y  + (eps/2)M^{-1/2}grad(log pi)(x)
    x'(x,y*)  = x  + eps M^{-1/2}y*
    y'(x',y*) = y* + (eps/2)M^{-1/2}grad(log pi)(x')
    """
    Q, P, S = size(z)

    # Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad) and full step position update
    momentum = copy(p)
    position_update!(z, p, ∇z, M, η)

    # Get gradient at new position
    logpos_ẑ, ∇ẑ, st_lux =
        logpos_withgrad(T.(z), T.(∇z), x, temps, model, ps, st_kan, st_lux)

    # Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad)
    momentum_update!(p, ∇ẑ, M, η)

    # Hamiltonian difference for transformed momentum
    # H(x,y) = -log(pi(x)) + (1/2)||p||^2 since p ~ N(0,I)
    log_r =
        logpos_ẑ - logpos_z -
        dropdims(
            sum(p .^ 2; dims = (1, 2)) - sum(momentum .^ 2; dims = (1, 2));
            dims = (1, 2),
        ) ./ 2

    return z, logpos_ẑ, ∇ẑ, -p, log_r, st_lux
end

end
