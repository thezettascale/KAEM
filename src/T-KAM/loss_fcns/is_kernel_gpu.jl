module IS_Kernel

export loss_accum

using ..Utils

using CUDA, KernelAbstractions

function accumulator(
    weights::AbstractArray{T,1},
    logprior::AbstractArray{T,1},
    logllhood::AbstractArray{T,1},
)::T where {T<:half_quant}
    return weights' * (logprior + logllhood)
end

function loss_accum(
    weights_resampled::AbstractArray{T,2},
    logprior::AbstractArray{T,1},
    logllhood::AbstractArray{T,2},
    resampled_idxs::AbstractArray{Int,2},
    B::Int,
    S::Int,
)::T where {T<:half_quant}

    loss = zero(T)
    for b in 1:B
        loss = loss + accumulator(
            weights_resampled[b, :],
            logprior[resampled_idxs[b, :]],
            logllhood[b, resampled_idxs[b, :]],
        )
    end

    return loss / B
end

end
