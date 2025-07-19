module WeightResamplers

export residual_resampler, systematic_resampler, stratified_resampler, importance_resampler

using CUDA, Random, Distributions, LinearAlgebra, ParallelStencil
using NNlib: softmax

include("../../utils.jl")
using .Utils: full_quant, device

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

@parallel_indices (b) function residual_kernel!(
    idxs::AbstractArray{U},
    ESS_bool::AbstractArray{Bool},
    cdf::AbstractArray{U},
    u::AbstractArray{U},
    num_remaining::AbstractArray{Int},
    integer_counts::AbstractArray{Int},
    B::Int,
    N::Int,
)::Nothing where {U<:full_quant}
    c = 1

    if !ESS_bool[b] # No resampling
        for n = 1:N
            idxs[b, n] = n
        end
    else

        # Deterministic replication as explicit assignment loop
        for s = 1:N
            count = integer_counts[b, s]
            if count > 0
                for i = c:(c+count-1)
                    idxs[b, i] = s
                end
                c += count
            end
        end

        # Multinomial resampling as explicit assignment loop
        if num_remaining[b] > 0
            for k = 1:num_remaining[b]
                idx = N
                for j = 1:N
                    if cdf[b, j] >= u[b, k]
                        idx = j
                        break
                    end
                end
                idx = idx > N ? N : idx
                idxs[b, c] = idx
                c += 1
            end
        end
    end
    return nothing
end

function residual_resampler(
    weights::AbstractArray{U},
    ESS_bool::AbstractArray{Bool},
    B::Int,
    N::Int;
    rng::AbstractRNG = Random.default_rng(),
)::AbstractArray{Int} where {U<:full_quant}
    """
    Residual resampling for weight filtering.

    Args:
        weights: The weights of the population.
        ESS_bool: A boolean array indicating if the ESS is above the threshold.
        rng: Random seed for reproducibility.

    Returns:
        - The resampled indices.
    """
    # Number times to replicate each sample, (convert to FP64 because stability issues)
    integer_counts = Int.(floor.(weights .* N))
    num_remaining = dropdims(N .- sum(integer_counts, dims = 2); dims = 2)

    # Residual weights to resample from
    residual_weights = softmax(weights .* (N .- integer_counts), dims = 2)

    # CDF and variate for resampling
    u = device(rand(rng, U, B, N))
    cdf = cumsum(residual_weights, dims = 2)

    idxs = @zeros(B, N)
    @parallel (1:B) residual_kernel!(
        idxs,
        ESS_bool,
        cdf,
        u,
        num_remaining,
        integer_counts,
        B,
        N,
    )
    return Int.(idxs)
end

@parallel_indices (b) function systematic_kernel!(
    idxs::AbstractArray{U},
    ESS_bool::AbstractArray{Bool},
    cdf::AbstractArray{U},
    u::AbstractArray{U},
    B::Int,
    N::Int,
)::Nothing where {U<:full_quant}
    if !ESS_bool[b] # No resampling
        for n = 1:N
            idxs[b, n] = n
        end
    else
        # Searchsortedfirst as explicit assignment loop
        for n = 1:N
            idx = N
            for j = 1:N
                if cdf[b, j] >= u[b, n]
                    idx = j
                    break
                end
            end
            idx = idx > N ? N : idx
            idxs[b, n] = idx
        end
    end
    return nothing
end

function systematic_resampler(
    weights::AbstractArray{U},
    ESS_bool::AbstractArray{Bool},
    B::Int,
    N::Int;
    rng::AbstractRNG = Random.default_rng(),
)::AbstractArray{Int} where {U<:full_quant}
    """
    Systematic resampling for weight filtering.

    Args:
        weights: The weights of the population.
        ESS_bool: A boolean array indicating if the ESS is above the threshold.
        rng: Random seed for reproducibility.

    Returns:
        - The resampled indices.
    """

    cdf = cumsum(weights, dims = 2)

    # Systematic thresholds
    u = device((rand(rng, U, B, 1) .+ (0:(N-1))') ./ N)

    idxs = @zeros(B, N)
    @parallel (1:B) systematic_kernel!(idxs, ESS_bool, cdf, u, B, N)
    return Int.(idxs)
end

function stratified_resampler(
    weights::AbstractArray{U},
    ESS_bool::AbstractArray{Bool},
    B::Int,
    N::Int;
    rng::AbstractRNG = Random.default_rng(),
)::AbstractArray{Int} where {U<:full_quant}
    """
    Systematic resampling for weight filtering.

    Args:
        weights: The weights of the population.
        ESS_bool: A boolean array indicating if the ESS is above the threshold.
        rng: Random seed for reproducibility.

    Returns:
        - The resampled indices.
    """

    cdf = cumsum(weights, dims = 2)

    # Stratified thresholds
    u = device((rand(rng, U, B, N) .+ (0:(N-1))') ./ N)

    idxs = @zeros(B, N)
    @parallel (1:B) systematic_kernel!(idxs, ESS_bool, cdf, u, B, N)
    return Int.(idxs)
end

function importance_resampler(
    weights::AbstractArray{U};
    rng::AbstractRNG = Random.default_rng(),
    ESS_threshold::U = full_quant(0.5),
    resampler::Function = systematic_sampler,
    verbose::Bool = false,
)::AbstractArray{Int} where {U<:full_quant}
    """
    Filter the latent variable for a index of the Steppingstone sum using residual resampling.

    Args:
        logllhood: A matrix of log-likelihood values.
        weights: The weights of the population.
        t_resample: The temperature at which the last resample occurred.
        t2: The temperature at which to update the weights.
        rng: Random seed for reproducibility.
        ESS_threshold: The threshold for the effective sample size.
        resampler: The resampling function.

    Returns:
        - The resampled indices.    
    """
    B, N = size(weights)

    # Check effective sample size
    ESS = dropdims(1 ./ sum(weights .^ 2, dims = 2); dims = 2)
    ESS_bool = ESS .< ESS_threshold*N

    # Only resample when needed 
    verbose && (any(ESS_bool) && println("Resampling!"))
    any(ESS_bool) && return resampler(weights, ESS_bool, B, N; rng = rng)
    return repeat(collect(1:N)', B, 1)
end

end
