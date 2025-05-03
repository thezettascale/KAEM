module Preconditioning

export init_mass_matrix, sample_momentum

using LinearAlgebra, Random, Distributions, Statistics

include("../../utils.jl")
using .Utils: next_rng, full_quant

abstract type Preconditioner end

struct IdentityPreconditioner <: Preconditioner end
struct DiagonalPreconditioner <: Preconditioner end

struct MixDiagonalPreconditioner{TR<:Real} <: Preconditioner
    p0::TR  # Proportion of zeros
    p1::TR  # Proportion of ones
    
    function MixDiagonalPreconditioner(p0::TR, p1::TR) where {TR<:Real}
        zero(TR) ≤ p0+p1 ≤ one(TR) || throw(ArgumentError("p0+p1 < 0 or p0+p1 > 1"))
        new{TR}(p0, p1)
    end
end

MixDiagonalPreconditioner() = MixDiagonalPreconditioner(1//3, 1//3)

# Default behavior - no preconditioning
function build_preconditioner!(
    dest::AbstractArray{T}, 
    ::IdentityPreconditioner,
    std_devs::AbstractArray{T}; 
    seed::Int=1
    ) where T
    fill!(dest, one(T))
    return dest
end

# Diagonal preconditioning
function build_preconditioner!(
    dest::AbstractArray{T}, 
    ::DiagonalPreconditioner,
    std_devs::AbstractArray{T}; 
    seed::Int=1
    ) where T
    @. dest = ifelse(iszero(std_devs), one(T), one(T) / std_devs)
    return dest
end

# Mixed diagonal preconditioning
function build_preconditioner!(
    dest::AbstractArray{T}, 
    prec::MixDiagonalPreconditioner,
    std_devs::AbstractArray{T}; 
    seed::Int=1
    ) where T
    seed, rng = next_rng(seed)
    u = rand(rng, T)
    
    if u ≤ prec.p0
        # Use inverse standard deviations
        @. dest = ifelse(iszero(std_devs), one(T), one(T) / std_devs)
    elseif u ≤ prec.p0 + prec.p1
        # Use identity
        fill!(dest, one(T))
    else
        # Random mixture
        seed, rng = next_rng(seed)
        mix = rand(rng, T)
        rmix = one(T) - mix
        @. dest = ifelse(iszero(std_devs), 
                        one(T), 
                        mix + rmix / std_devs)
    end
    return dest
end

function init_mass_matrix(
    z::AbstractArray{full_quant},
    seed::Int=1,
    )
    Q, P, S = size(z)
    Σ = diag(cov(reshape(z, Q*P, S)'))

    seed, rng = next_rng(seed)
    β = rand(rng, Truncated(Beta(1, 1), 0.5, 2/3)) |> full_quant

    Σ_AM = sqrt.(β .* (1 ./ Σ) .+ (1 - β)) 
    return reshape(Σ_AM, Q, P), seed
end

# This is transformed momentum!
function sample_momentum(
    z::AbstractArray{full_quant},
    M::AbstractArray{full_quant};
    seed::Int=1,
    preconditioner::Preconditioner=MixDiagonalPreconditioner(),
    )
    Q, P, S = size(z)
    
    # Compute M^{1/2}
    z_reshaped = reshape(z, Q*P, S)
    μ = mean(z_reshaped, dims=2)
    Σ = sqrt.(@views sum((z_reshaped .- μ).^2, dims=2) ./ (S-1))
    
    # Initialize mass matrix (M^{1/2})
    build_preconditioner!(reshape(M, Q*P), preconditioner, vec(Σ); seed=seed)
    
    # Sample y ~ N(0,I) directly (transformed momentum)
    seed, rng = next_rng(seed)
    y = randn(rng, full_quant, Q, P, S)
    
    return y, M, seed
end 

end