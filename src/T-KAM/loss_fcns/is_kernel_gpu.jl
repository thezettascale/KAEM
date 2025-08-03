module IS_Kernel

export loss_accum

using ..Utils

using CUDA, KernelAbstractions, Tullio

function loss_accum(
    weights_resampled::AbstractArray{T,2},
    logprior::AbstractArray{T,1},
    logllhood::AbstractArray{T,2},
    resampled_idxs::AbstractArray{Int,2},
    B::Int,
    S::Int,
)::AbstractArray{T,1} where {T<:half_quant}
    lp = reduce(hcat, map(b -> logprior[resampled_idxs[b, :], :], 1:B))
    ll = reduce(vcat, map(b -> logllhood[b:b, resampled_idxs[b, :]], 1:B))
    @tullio lp_loss[b] := weights_resampled[b, s] * lp[s, b]
    @tullio ll_loss[b] := weights_resampled[b, s] * ll[b, s]
    return lp_loss + ll_loss
end

end
