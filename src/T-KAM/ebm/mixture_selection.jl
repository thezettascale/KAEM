
module MixtureChoice

export choose_component

using NNlib: softmax
using Flux: onehotbatch
using CUDA, LinearAlgebra, Random, ParallelStencil

include("../../utils.jl")
using .Utils: pu, half_quant, full_quant, fq

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

@parallel_indices (q, b) function mask_kernel!(
    mask::AbstractArray{U},
    α::AbstractArray{U},
    rand_vals::AbstractArray{U},
    p_size::Int,
)::Nothing where {U<:full_quant}
    idx = p_size
    val = rand_vals[q, b]

    # Potential thread divergence on GPU
    for j = 1:p_size
        if α[q, j] >= val
            idx = j
            break
        end
    end

    # One-hot vector for this (q, b)
    for k = 1:p_size
        mask[q, b, k] = (idx == k) ? one(U) : zero(U)
    end
    return nothing
end

function choose_component(
    α::AbstractArray{half_quant},
    num_samples::Int,
    q_size::Int,
    p_size::Int;
    rng::AbstractRNG = Random.default_rng(),
)::AbstractArray{full_quant}
    """
    Creates a one-hot mask for mixture model, q, to select one component, p.

    Args:
        alpha: The mixture proportions, (q, p).
        num_samples: The number of samples to generate.
        q_size: The number of mixture models.
        rng: The random number generator.

    Returns:
        chosen_components: The one-hot mask for each mixture model, (num_samples, q, p).    
    """
    rand_vals = rand(rng, full_quant, q_size, num_samples)
    α = cumsum(softmax(full_quant.(α); dims = 2); dims = 2)

    mask = @zeros(q_size, p_size, num_samples)
    @parallel (1:q_size, 1:num_samples) mask_kernel!(mask, α, rand_vals, p_size)
    return mask
end

end
