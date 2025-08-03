module IS_Kernel

export loss_accum

using ..Utils

using CUDA, ParallelStencil

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, half_quant, 3)
else
    @init_parallel_stencil(Threads, half_quant, 3)
end

@parallel_indices (b, s) function resampled_kernel!(
    loss::AbstractArray{T,2},
    weights_resampled::AbstractArray{T,2},
    logprior::AbstractArray{T,1},
    logllhood::AbstractArray{T,2},
    resampled_idxs::AbstractArray{Int,2},
)::Nothing where {T<:half_quant}
    idx = resampled_idxs[b, s]
    loss[b, s] = weights_resampled[b, s] * (logprior[idx] + logllhood[b, idx])
    return nothing
end

function loss_accum(
    weights_resampled::AbstractArray{T,2},
    logprior::AbstractArray{T,1},
    logllhood::AbstractArray{T,2},
    resampled_idxs::AbstractArray{Int,2},
    B::Int,
    S::Int,
)::AbstractArray{T,1} where {T<:half_quant}
    marginal_llhood = @zeros(B, S)
    @parallel (1:B, 1:S) resampled_kernel!(
        marginal_llhood,
        weights_resampled,
        logprior,
        logllhood,
        resampled_idxs,
    )
    return dropdims(sum(marginal_llhood; dims = 2); dims = 2)
end

end
