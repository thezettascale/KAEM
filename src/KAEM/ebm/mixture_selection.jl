module MixtureChoice

export choose_component

using NNlib: softmax
using LinearAlgebra, Lux

using ..Utils

function mask_kernel(
        α,
        rand_vals,
        Q,
        P
    )
    """One-hot mask for chosen index"""
    indices = sum(α .< rand_vals; dims = 2) .+ 1
    p_range = reshape(1:P, 1, P, 1)
    mask = indices .== p_range |> Lux.f32
    return mask
end

function choose_component(
        α,
        num_samples,
        q_size,
        p_size,
        st_rng;
        ula_init = false
    )
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
    rand_vals = ula_init ? st_rng.mix_rv_ula : st_rng.mix_rv
    α = cumsum(softmax(α; dims = 2); dims = 2)
    return mask_kernel(α, rand_vals, q_size, p_size)
end

end
