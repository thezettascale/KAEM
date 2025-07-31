module IS_Kernel

export loss_accum

using ..Utils

using CUDA, KernelAbstractions, Tullio

function loss_accum(
    weights_resampled::AbstractArray{T},
    logprior::AbstractArray{T},
    logllhood::AbstractArray{T},
    resampled_idxs::AbstractArray{Int},
    B::Int,
    S::Int,
)::AbstractArray{T} where {T<:half_quant}
    lp = reduce(hcat, map(b -> logprior[resampled_idxs[b, :], :], 1:B))
    ll = reduce(vcat, map(b -> logllhood[b:b, resampled_idxs[b, :]], 1:B))
    @tullio out[b] := weights_resampled[b, s] * (lp[s, b] + ll[b, s])
    return out
end

end
