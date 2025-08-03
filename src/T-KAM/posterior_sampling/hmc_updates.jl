module HamiltonianMonteCarlo

export leapfrog, logpos_withgrad

using CUDA, Lux, LuxCUDA, ComponentArrays, Accessors, ParallelStencil
using Enzyme: make_zero

using ..Utils
using ..T_KAM_model

include("log_posteriors.jl")
using .LogPosteriors: autoMALA_value_and_grad

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

@parallel_indices (q, p, s) function position_update!(
    z::AbstractArray{U,3},
    p::AbstractArray{U,3},
    ∇z::AbstractArray{U,3},
    M::AbstractArray{U,2},
    η::AbstractArray{U,1},
)::Nothing where {U<:full_quant}
    p[q, p, s] = p[q, p, s] + (η[s] / 2) * ∇z[q, p, s] / M[q, p]
    z[q, p, s] = z[q, p, s] + η[s] * p[q, p, s] / M[q, p]
    return nothing
end

@parallel_indices (q, p, s) function momentum_update!(
    p::AbstractArray{U,3},
    ∇ẑ::AbstractArray{U,3},
    M::AbstractArray{U,2},
    η::AbstractArray{U,1},
)::Nothing where {U<:full_quant}
    p[q, p, s] = p[q, p, s] + (η[s] / 2) * ∇ẑ[q, p, s] / M[q, p]
    return nothing
end

function logpos_withgrad(
    z::AbstractArray{U,3},
    ∇z::AbstractArray{U,3},
    x::AbstractArray{T},
    temps::AbstractArray{T,1},
    model,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::Tuple{
    AbstractArray{U,1},
    AbstractArray{U,3},
    NamedTuple,
} where {T<:half_quant,U<:full_quant}
    logpos, ∇z_k, st_ebm, st_gen = autoMALA_value_and_grad(z, ∇z, x, temps, model, ps, st_kan, st_lux)
    @reset st_lux.ebm = st_ebm
    @reset st_lux.gen = st_gen

    return U.(logpos) ./ U(model.loss_scaling),
    U.(∇z_k) ./ U(model.loss_scaling),
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
    # # Half-step momentum update (p* = p + (eps/2)M^{-1/2}grad) and full step position update
    momentum = copy(p)
    position_update!(z, p, ∇z, M, η)

    # Get gradient at new position
    logpos_ẑ, ∇ẑ, st_lux = logpos_withgrad(T.(z), T.(∇z), x, temps, model, ps, st_kan, st_lux)

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

    return ẑ, logpos_ẑ, ∇ẑ, -p, log_r, st_lux
end

end
