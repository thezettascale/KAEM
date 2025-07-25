module HamiltonianMonteCarlo

export leapfrog, logpos_withgrad

using CUDA, Lux, LuxCUDA, ComponentArrays, Accessors
using Enzyme: make_zero

using ..Utils
using ..T_KAM_model

include("log_posteriors.jl")
using .LogPosteriors: autoMALA_value_and_grad_4D, autoMALA_value_and_grad

function position_update(
    z::AbstractArray{U},
    momentum::AbstractArray{U},
    ∇z::AbstractArray{U},
    M::AbstractArray{U},
    η::AbstractArray{U},
)::Tuple{AbstractArray{U},AbstractArray{U}} where {U<:full_quant}
    ηr = reshape(η, 1, 1, size(η)...)
    y = @. momentum + (ηr / 2) * ∇z / M
    @. z = z + ηr * y / M
    return y, z
end

function momentum_update!(
    y::AbstractArray{U},
    ∇ẑ::AbstractArray{U},
    M::AbstractArray{U},
    η::AbstractArray{U},
)::Nothing where {U<:full_quant}
    ηr = reshape(η, 1, 1, size(η)...)
    @. y + (ηr / 2) * ∇ẑ / M
    return nothing
end

function logpos_withgrad(
    z::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    model,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::Tuple{
    AbstractArray{full_quant},
    AbstractArray{full_quant},
    NamedTuple,
} where {T<:half_quant}
    fcn = ndims(z) == 4 ? autoMALA_value_and_grad_4D : autoMALA_value_and_grad
    logpos, ∇z_k, st_ebm, st_gen = fcn(z, make_zero(z), x, temps, model, ps, st_kan, st_lux)
    @reset st_kan.ebm = st_ebm
    @reset st_lux.gen = st_gen

    return full_quant.(logpos) ./ full_quant(model.loss_scaling),
    full_quant.(∇z_k) ./ full_quant(model.loss_scaling),
    st_kan,
    st_lux
end

function leapfrog(
    z::AbstractArray{U},
    ∇z::AbstractArray{U},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    logpos_z::AbstractArray{U},
    momentum::AbstractArray{U},  # This is y = M^{-1/2}p
    M::AbstractArray{U},         # This is M^{1/2}
    η::AbstractArray{U},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
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
    logpos_ẑ, ∇ẑ, st_kan, st_lux =
        logpos_withgrad(T.(ẑ), x, temps, model, ps, st_kan, st_lux)

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

    return ẑ, logpos_ẑ, ∇ẑ, -p, log_r, st_kan, st_lux
end

end
