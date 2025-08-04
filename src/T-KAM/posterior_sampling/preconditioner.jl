module Preconditioning

export init_mass_matrix, sample_momentum

using LinearAlgebra, Random, Distributions, Statistics

using ..Utils

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
    dest::AbstractArray{U,2},
    ::IdentityPreconditioner,
    std_devs::AbstractArray{U,2};
    rng::AbstractRNG = Random.default_rng(),
)::Nothing where {U<:full_quant}
    fill!(dest, one(U))
    return nothing
end

# Diagonal preconditioning
function build_preconditioner!(
    dest::AbstractArray{U,2},
    ::DiagonalPreconditioner,
    std_devs::AbstractArray{U,2};
    rng::AbstractRNG = Random.default_rng(),
)::Nothing where {U<:full_quant}
    @. dest = ifelse(iszero(std_devs), one(U), one(U) / std_devs)
    return nothing
end

# Mixed diagonal preconditioning
function build_preconditioner!(
    dest::AbstractArray{U,2},
    prec::MixDiagonalPreconditioner,
    std_devs::AbstractArray{U,2};
    rng::AbstractRNG = Random.default_rng(),
)::Nothing where {U<:full_quant}
    u = rand(rng, U)

    if u ≤ prec.p0
        # Use inverse standard deviations
        @. dest = ifelse(iszero(std_devs), one(U), one(U) / std_devs)
    elseif u ≤ prec.p0 + prec.p1
        # Use identity
        fill!(dest, one(U))
    else
        # Random mixture
        mix = rand(rng, U)
        rmix = one(U) - mix
        @. dest = ifelse(iszero(std_devs), one(U), mix + rmix / std_devs)
    end
    return nothing
end

function init_mass_matrix(
    z::AbstractArray{U,3},
    rng::AbstractRNG = Random.default_rng(),
)::AbstractArray{U,2} where {U<:full_quant}
    Σ = sum((z .- mean(z; dims = 3)) .^ 2; dims = 3) ./ (size(z, 3) - 1) # Diagonal Covariance
    β = rand(rng, Truncated(Beta(1, 1), 0.5, 2/3)) |> U
    @. Σ = sqrt(β * (1 / Σ) + (1 - β)) # Augmented mass matrix 
    return dropdims(Σ; dims = 3)
end

# This is transformed momentum!
function sample_momentum(
    z::AbstractArray{U,3},
    M::AbstractArray{U,2};
    rng::AbstractRNG = Random.default_rng(),
    preconditioner::Preconditioner = MixDiagonalPreconditioner(),
)::Tuple{AbstractArray{U,3},AbstractArray{U,2}} where {U<:full_quant}

    # Initialize M^{1/2}
    Σ = sqrt.(sum((z .- mean(z; dims = 3)) .^ 2; dims = 3) ./ (size(z, 3) - 1))
    build_preconditioner!(M, preconditioner, dropdims(Σ; dims = 3); rng = rng)

    # Sample y ~ N(0,I) directly (transformed momentum)
    y = randn(rng, U, size(z)) |> pu
    return y, M
end

end
