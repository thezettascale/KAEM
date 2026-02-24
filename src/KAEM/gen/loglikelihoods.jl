module LogLikelihoods

export log_likelihood_IS, log_likelihood_MALA

using ComponentArrays, Random
using NNlib: softmax, sigmoid

using ..Utils
using ..KAEM_model: GenModel

include("losses.jl")
using .Losses

## Log-likelihood functions ##
function log_likelihood_IS(
        z,
        x,
        lkhood,
        ps,
        st_kan,
        st_lux,
        noise;
        ε = eps(Float32),
    )
    """Importance-sampled conditional log-likelihood."""
    x̂, st_lux_new = lkhood.generator(ps, st_kan, st_lux, z)
    noise_scaled = lkhood.σ.noise .* noise
    x̂_noised = lkhood.output_activation(x̂ .+ noise_scaled)

    # Add singleton dimension
    x_expanded = (
        length(lkhood.x_shape) == 3 ?
            PermutedDimsArray(view(x, :, :, :, :, :), (1, 2, 3, 5, 4)) :
            (
                length(lkhood.x_shape) == 2 ?
                PermutedDimsArray(view(x, :, :, :, :), ((1, 2, 4, 3))) :
                PermutedDimsArray(view(x, :, :, :), ((1, 3, 2)))
            )
    )

    ll = IS_loss(x_expanded, x̂_noised, ε, 2 * lkhood.σ.llhood^2, lkhood.SEQ)
    return ll, st_lux_new
end

function log_likelihood_MALA(
        z,
        x,
        lkhood,
        ps,
        st_kan,
        st_lux,
        noise;
        ε = eps(Float32),
    )
    """MALA-sampled conditional log-likelihood."""
    x̂, st_lux_new = lkhood.generator(ps, st_kan, st_lux, z)
    noise_scaled = lkhood.σ.noise .* noise
    x̂_act = lkhood.output_activation(x̂ .+ noise_scaled)

    ll =
        MALA_loss(x, x̂_act, ε, 2 * lkhood.σ.llhood^2, lkhood.SEQ, lkhood.perceptual_scale)
    return ll, st_lux_new
end

end
