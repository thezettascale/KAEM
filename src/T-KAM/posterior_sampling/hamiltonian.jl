module HamiltonianDynamics

export position_update, momentum_update, leapfrop_proposal

using CUDA, KernelAbstractions, Tullio, Lux, LuxCUDA, ComponentArrays, ParallelStencil

include("../../utils.jl")
using .Utils: full_quant, half_quant

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 4)
else
    @init_parallel_stencil(Threads, full_quant, 4)
end

# Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad) and full step position update
@parallel_indices (q, p, s, t) function position_update!(
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

# Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad)
@parallel_indices (q, p, s, t) function momentum_update!(
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
    Q, P, S = size(z)[1:3]...

    if ndims(z) == 3
        p = @zeros(Q, P, S, 1)
        ẑ = @zeros(Q, P, S, 1)
        @parallel (1:Q, 1:P, 1:S, 1:1) position_update!(p, ẑ, momentum[:,:,:,:], ∇z[:,:,:,:], M[:,:,:,:], η[:,:])
        logpos_ẑ, ∇ẑ, st_kan, st_lux = logpos_withgrad(T.(ẑ), x, temps, model, ps, st_kan, st_lux)
        @parallel (1:Q, 1:P, 1:S, 1:1) momentum_update!(p, ∇ẑ, M[:,:,:,:], η[:,:])
        ẑ, logpos_ẑ, ∇ẑ, -p = ẑ[:,:,:,1], logpos_ẑ[:,1], ∇ẑ[:,:,:,1], p[:,:,:,1]
    else
        num_temps = size(η, 2)
        p = @zeros(Q, P, S, num_temps)
        ẑ = @zeros(Q, P, S, num_temps)
        @parallel (1:Q, 1:P, 1:S, 1:num_temps) position_update!(p, ẑ, momentum, ∇z, M, η)
        logpos_ẑ, ∇ẑ, st_kan, st_lux = logpos_withgrad(T.(ẑ), x, temps, model, ps, st_kan, st_lux)
        @parallel (1:Q, 1:P, 1:S, 1:num_temps) momentum_update!(p, ∇ẑ, M, η)
    end

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
