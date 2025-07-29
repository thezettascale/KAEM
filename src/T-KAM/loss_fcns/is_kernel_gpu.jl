module IS_Kernel

export loss_accum

using ..Utils

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
    return weights_resampled .* (lp' .+ ll)
end

end
