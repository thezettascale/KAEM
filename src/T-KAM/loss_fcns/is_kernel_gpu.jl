module IS_Kernel

export loss_accum

using ..Utils

using CUDA, KernelAbstractions, Tullio

function accumulator(
    weights::AbstractArray{T,1},
    logprior::AbstractArray{T,1},
    logllhood::AbstractArray{T,2},
    resampled_idxs::AbstractArray{Int,2},
)::T where {T<:half_quant}
    return @tullio loss := weights[s] * (logprior[s] + logllhood[s])
end

function loss_accum(
    weights_resampled::AbstractArray{T,2},
    logprior::AbstractArray{T,1},
    logllhood::AbstractArray{T,2},
    resampled_idxs::AbstractArray{Int,2},
    B::Int,
    S::Int,
)::T where {T<:half_quant}

    loss = reduce(
        mean,
        map(
            b -> accumulator(
                weights_resampled[b, :],
                logprior[resampled_idxs[b, :]],
                logllhood[b, resampled_idxs[b, :]],
            ),
            1:B,
        ),
    )

    return loss
end

end
