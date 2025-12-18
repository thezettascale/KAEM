module WeightResamplers

export ResidualResampler, SystematicResampler, StratifiedResampler, resampler_map

using LinearAlgebra, Lux
using NNlib: softmax
using Reactant: @allowscalar

using ..Utils

### Note: potential thread divergence on GPU for resampler searchsortedfirsts
function check_ESS(
        weights;
        ESS_threshold = 0.5f0,
    )
    """Effective sample size"""
    B, N = size(weights)
    ESS = 1 ./ sum(weights .^ 2, dims = 2)
    ESS_bool = ESS .< ESS_threshold * N
    return ESS_bool, B, N
end

struct ResidualResampler <: AbstractResampler
    ESS_threshold::Float32
    _phantom::Bool
end

function residual_kernel(
        ESS_bool,
        cdf,
        u,
        integer_counts,
        num_remaining,
        B,
        N,
    )
    early_return = (1 .- ESS_bool) .* (1:N)'
    mask = (1:N)' .>= (N .- num_remaining .+ 1) # Whether allcoated residual or not

    # Replicate by integer_counts in a stableHLO-compatible manner
    c = cumsum(integer_counts; dims = 2)
    deterministic_part = 1 .+ dropdims(
        sum(
            c .< reshape(1:N, 1, 1, N); dims = 2
        ); dims = 2
    )

    # Fill remaining with multinomial sampling
    residual_part = dropdims(
        sum(
            cdf .< u; dims = 2
        ) .+ 1; dims = 2
    )
    residual_part = ifelse.(residual_part .> N, N, residual_part)

    indices = (mask .* residual_part) .+ (1 .- mask) .* deterministic_part
    return early_return .+ ESS_bool .* indices
end

function (r::ResidualResampler)(
        weights,
        st_rng
    )
    """
    Residual resampling for weight filtering.

    Args:
        weights: The weights of the population.
        ESS_bool: A boolean array indicating if the ESS is above the threshold.
        rng: Random seed for reproducibility.

    Returns:
        - The resampled indices.
    """
    ESS_bool, B, N = check_ESS(
        weights;
        ESS_threshold = r.ESS_threshold,
    )

    # Number times to replicate each sample
    integer_counts = floor.(weights .* N)
    R = N .- sum(integer_counts, dims = 2)

    # Residual weights to resample from
    residual_mass = weights .* N .- integer_counts
    zero_vec = zero(residual_mass)
    residual_weights = ifelse.(
        R .> zero_vec,
        residual_mass ./ R,
        zero_vec
    )

    # CDF and variate for resampling
    u = st_rng.resample_rv[:, 1:1, :]
    cdf = cumsum(residual_weights, dims = 2)
    return residual_kernel(
        ESS_bool,
        cdf,
        u,
        integer_counts,
        dropdims(R; dims = 2),
        B,
        N
    )
end

struct SystematicResampler <: AbstractResampler
    ESS_threshold::Float32
    _phantom::Bool
end

function systematic_kernel(
        ESS_bool,
        cdf,
        u,
        B,
        N,
    )
    early_return = (1 .- ESS_bool) .* (1:N)'
    indices = dropdims(
        sum(
            cdf .< PermutedDimsArray(u, (1, 3, 2)); dims = 2
        ) .+ 1; dims = 2
    )
    indices = ifelse.(indices .> N, N, indices)
    return early_return .+ ESS_bool .* indices
end

function (r::SystematicResampler)(
        weights,
        st_rng
    )
    """
    Systematic resampling for weight filtering.

    Args:
        weights: The weights of the population.
        ESS_bool: A boolean array indicating if the ESS is above the threshold.
        rng: Random seed for reproducibility.

    Returns:
        - The resampled indices.
    """
    ESS_bool, B, N = check_ESS(weights; ESS_threshold = r.ESS_threshold)

    cdf = cumsum(weights, dims = 2)

    # Systematic thresholds
    rv = PermutedDimsArray(st_rng.resample_rv, (1, 3, 2))
    u = (rv .+ (0:(N - 1))') ./ N
    return systematic_kernel(ESS_bool, cdf, u, B, N)
end

struct StratifiedResampler <: AbstractResampler
    ESS_threshold::Float32
    _phantom::Bool
end

function (r::StratifiedResampler)(
        weights,
        st_rng
    )
    """
    Systematic resampling for weight filtering.

    Args:
        weights: The weights of the population.
        ESS_bool: A boolean array indicating if the ESS is above the threshold.
        rng: Random seed for reproducibility.

    Returns:
        - The resampled indices.
    """
    ESS_bool, B, N = check_ESS(weights; ESS_threshold = r.ESS_threshold)

    cdf = cumsum(weights, dims = 2)

    # Stratified thresholds
    rv = st_rng.resample_rv
    u = (rv .+ (0:(N - 1))') ./ N
    return systematic_kernel(ESS_bool, cdf, u, B, N)
end

struct NoResampler <: AbstractResampler
end

function (r::NoResampler)(
        weights,
        st_rng
    )
    B, N = size(weights)
    return repeat((1:N)', B, 1)
end

const resampler_map::Dict{String, Function} = Dict(
    "residual" => (ESS_threshold, verbose) -> ResidualResampler(ESS_threshold, verbose),
    "systematic" => (ESS_threshold, verbose) -> SystematicResampler(ESS_threshold, verbose),
    "stratified" => (ESS_threshold, verbose) -> StratifiedResampler(ESS_threshold, verbose),
    "none" => (ESS_threshold, verbose) -> NoResampler(),
)
end
