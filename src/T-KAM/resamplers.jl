module ParticleFilterResamplers

export residual_resampler, systematic_resampler

using Random, Distributions, LinearAlgebra
using NNlib: softmax

include("../utils.jl")
using .Utils: next_rng, quant

function residual_resampler(weights::AbstractArray{quant}, ESS_bool::AbstractArray{Bool}, B::Int, N::Int; seed::Int=1)
    """
    Residual resampling for particle filtering.

    Args:
        weights: The weights of the particles.
        ESS_bool: A boolean array indicating if the ESS is above the threshold.
        seed: Random seed for reproducibility.

    Returns:
        - The resampled indices.
        - The updated seed.
    """
    # Number times to replicate each particle
    integer_counts = floor.(weights .* N) .|> Int
    num_remaining = dropdims(N .- sum(integer_counts, dims=2); dims=2)

    # Residual weights to resample from
    residual_weights = softmax(weights .- (integer_counts ./ N), dims=2)

    # CDF and variate for resampling
    seed, rng = next_rng(seed)
    u = rand(rng, quant, B, maximum(num_remaining))
    cdf = cumsum(residual_weights, dims=2)

    idxs = Array{Int}(undef, B, N)
    Threads.@threads for b in 1:B
        c = 1

        if ESS_bool[b]
            idxs[b, :] .= 1:N
            continue
        end

        # Deterministic replication
        for s in 1:N
            count = integer_counts[b, s]
            if count > 0
                idxs[b, c:c+count-1] .= s
                c += count
            end
        end

        # Multinomial resampling
        if num_remaining[b] > 0
            idxs[b, c:end] .= searchsortedfirst.(Ref(cdf[b, :]), u[b, 1:num_remaining[b]])
        end
    end
    replace!(idxs, N+1 => N)

    return idxs, seed
end

function systematic_resampler(weights::AbstractArray{quant}, ESS_bool::AbstractArray{Bool}, B::Int, N::Int; seed::Int=1)
    """
    Systematic resampling for particle filtering.

    Args:
        weights: The weights of the particles.
        ESS_bool: A boolean array indicating if the ESS is above the threshold.
        seed: Random seed for reproducibility.

    Returns:
        - The resampled indices.
        - The updated seed.
    """

    cdf = cumsum(weights, dims=2)

    # Systematic thresholds
    seed, rng = next_rng(seed)
    u = (rand(rng, quant, B, 1) .+ (0:N-1)') ./ N

    idxs = Array{Int}(undef, B, N)
    Threads.@threads for b in 1:B
        if ESS_bool[b]
            idxs[b, :] .= 1:N
            continue
        end

        idxs[b, :] .= searchsortedfirst.(Ref(cdf[b, :]), u[b, :])
    end
    replace!(idxs, N+1 => N)

    return idxs, seed
end

end