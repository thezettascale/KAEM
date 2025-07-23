module HamiltonianDynamics

export position_update, momentum_update, leapfrop_proposal

using CUDA, KernelAbstractions, Tullio, Lux, LuxCUDA, ComponentArrays, ParallelStencil

include("../../utils.jl")
using .Utils: full_quant, half_quant

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

@parallel_indices (q, p, s) function position_update_3D!(
    p::AbstractArray{U},
    ẑ::AbstractArray{U},
    momentum::AbstractArray{U},
    ∇z::AbstractArray{U},
    M::AbstractArray{U},
    η::AbstractArray{U},
)::Nothing where {U<:full_quant}
    p[q, p, s] = momentum[q, p, s] + (η[s] / 2) * ∇z[q, p, s] / M[q, p, s]
    ẑ[q, p, s] = ẑ[q, p, s] + η[s] * p[q, p, s] / M[q, p, s]
    return nothing
end

@parallel_indices (q, p, s, t) function position_update_4D!(
    p::AbstractArray{U},
    ẑ::AbstractArray{U},
    momentum::AbstractArray{U},
    ∇z::AbstractArray{U},
    M::AbstractArray{U},
    η::AbstractArray{U},
)::Nothing where {U<:full_quant}
    p[q, p, s, t] = momentum[q, p, s, t] + (η[s, t] / 2) * ∇z[q, p, s, t] / M[q, p, s, t]
    ẑ[q, p, s, t] = ẑ[q, p, s, t] + η[s, t] * p[q, p, s, t] / M[q, p, s, t]
    return nothing
end

@parallel_indices (q, p, s) function momentum_update_3D!(
    p::AbstractArray{U},
    ∇ẑ::AbstractArray{U},
    M::AbstractArray{U},
    η::AbstractArray{U},
)::Nothing where {U<:full_quant}
    p[q, p, s] = p[q, p, s] + (η[s] / 2) * ∇ẑ[q, p, s] / M[q, p, s]
    return nothing
end

@parallel_indices (q, p, s, t) function momentum_update_4D!(
    p::AbstractArray{U},
    ∇ẑ::AbstractArray{U},
    M::AbstractArray{U},
    η::AbstractArray{U},
)::Nothing where {U<:full_quant}
    p[q, p, s, t] = p[q, p, s, t] + (η[s, t] / 2) * ∇ẑ[q, p, s, t] / M[q, p, s, t]
    return nothing
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
    num_temps = ndims(η) == 1 ? 1 : size(η, 2)
    Q, P, S = size(z)[1:3]...
    p = @zeros(size(z))
    ẑ = @zeros(size(z))

    # # Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad) and full step position update
    if ndims(z) == 3
        @parallel (1:Q, 1:P, 1:S) position_update_3D!(p, ẑ, momentum, ∇z, M, η)
    else
        @parallel (1:Q, 1:P, 1:S, 1:num_temps) position_update_4D!(p, ẑ, momentum, ∇z, M, η)
    end

    # Get gradient at new position
    logpos_ẑ, ∇ẑ, st = logpos_withgrad(T.(ẑ), x, temps, model, ps, st_kan, st_lux)

    # Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad)
    if ndims(z) == 3
        @parallel (1:Q, 1:P, 1:S) momentum_update_3D!(p, ∇ẑ, M, η)
    else
        @parallel (1:Q, 1:P, 1:S, 1:num_temps) momentum_update_4D!(p, ∇ẑ, M, η)
    end

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
