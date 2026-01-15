module PangEBMSampling

export seed_pang_rng, langevin_prior, langevin_posterior, generate_pang

using Lux, Random, Enzyme

using ..Utils
using ..PangEBMModel

# Pre-generate all RNG
function seed_pang_rng(
        model::PangEBM{T};
        rng::AbstractRNG = Random.MersenneTwister(1),
        batch_size::Int = model.batch_size,
    ) where {T <: Float32}

    latent_dim = model.latent_dim
    prior_steps = model.prior_sgld_steps
    post_steps = model.post_sgld_steps

    return (
        prior_init = randn(rng, T, latent_dim, batch_size),
        prior_noise = randn(rng, T, latent_dim, batch_size, prior_steps),
        post_init = randn(rng, T, latent_dim, batch_size),
        post_noise = randn(rng, T, latent_dim, batch_size, post_steps),
    ) |> pu
end

function neg_energy_sum(z, energy_net, ps_ebm, st_ebm)
    E, _ = energy_net(z, ps_ebm, st_ebm)
    return -sum(E)  # log p(z) = -E(z)
end

# log p(z|x) = log p(x|z) + log p(z) = log p(x|z) - E(z)
function log_posterior_sum(z, x, model, ps, st, σ²)
    E, _ = model.energy_net(z, ps.ebm, st.ebm)
    ll, _ = log_likelihood(model, x, z, ps, st; σ² = σ²)
    return sum(ll) - sum(E)
end

function langevin_prior(model::PangEBM, ps, st, st_rng)
    z = st_rng.prior_init
    η = model.prior_sgld_step_size
    σ = model.noise_scale
    σ_prior² = model.prior_sigma^2
    sqrt_2η = sqrt(2.0f0 * η)

    for step in 1:model.prior_sgld_steps
        ∇z = first(
            Enzyme.gradient(
                Enzyme.Reverse,
                Enzyme.Const(neg_energy_sum),
                z,
                Enzyme.Const(model.energy_net),
                Enzyme.Const(ps.ebm),
                Enzyme.Const(st.ebm),
            )
        )

        # Gaussian prior gradient: ∇ log p_gaussian(z) = -z/σ_prior²
        ∇z_prior = -z ./ σ_prior²

        ε = st_rng.prior_noise[:, :, step]
        z = z .+ η .* (∇z .+ ∇z_prior) .+ σ .* sqrt_2η .* ε
    end

    return z
end

# ULA for posterior: sample z ~ p(z|x) ∝ p(x|z)p(z)
function langevin_posterior(model::PangEBM, x, ps, st, st_rng)
    σ² = model.likelihood_variance
    σ_prior² = model.prior_sigma^2
    z = st_rng.post_init
    η = model.post_sgld_step_size
    σ = model.noise_scale
    sqrt_2η = sqrt(2.0f0 * η)

    for step in 1:model.post_sgld_steps
        ∇z = first(
            Enzyme.gradient(
                Enzyme.Reverse,
                Enzyme.Const(log_posterior_sum),
                z,
                Enzyme.Const(x),
                Enzyme.Const(model),
                Enzyme.Const(ps),
                Enzyme.Const(st),
                Enzyme.Const(σ²),
            )
        )

        # Gaussian ref: ∇ log p_gaussian(z) = -z/σ_prior²
        ∇z_prior = -z ./ σ_prior²

        ε = st_rng.post_noise[:, :, step]
        z = z .+ η .* (∇z .+ ∇z_prior) .+ σ .* sqrt_2η .* ε
    end

    return z
end

function generate_pang(model::PangEBM, ps, st, st_rng)
    z = langevin_prior(model, ps, st, st_rng)
    x_gen, st_new = model(ps, st, z)
    return x_gen, st_new
end

end
