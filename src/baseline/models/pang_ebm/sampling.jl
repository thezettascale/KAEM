module PangEBMSampling

export langevin_prior, langevin_posterior, generate_pang

using Lux, Enzyme
using Reactant: @trace

using ..Utils
using ..PangEBMModel

function neg_energy_sum(z, energy_net, ps_ebm, st_ebm)
    E, _ = energy_net(z, ps_ebm, st_ebm)
    return -sum(E)  # log p(z) = -E(z)
end

# log p(z|x) = log p(x|z) + log p(z) = log p(x|z) - E(z)
function log_posterior_sum(z, x, model, ps, st)
    E, _ = model.energy_net(z, ps.ebm, st.ebm)
    ll, _ = log_likelihood(model, x, z, ps, st)
    return sum(ll) - sum(E)
end

function prior_step(
        i,
        z,
        energy_net,
        ps_ebm,
        st_ebm,
        η,
        σ,
        sqrt_2η,
        σ_prior²,
        prior_noise,
    )
    ∇z = first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(neg_energy_sum),
            z,
            Enzyme.Const(energy_net),
            Enzyme.Const(ps_ebm),
            Enzyme.Const(st_ebm),
        )
    )

    # Gaussian prior gradient: ∇ log p_gaussian(z) = -z/σ_prior²
    ∇z_prior = -z ./ σ_prior²
    ε = prior_noise[:, :, i]
    z_new = z .+ η .* (∇z .+ ∇z_prior) .+ σ .* sqrt_2η .* ε
    return z_new
end

function langevin_prior(model::PangEBM, ps, st, st_rng)
    z_init = st_rng.prior_init
    η = model.prior_sgld_step_size
    σ = model.noise_scale
    σ_prior² = model.prior_sigma^2
    sqrt_2η = sqrt(2.0f0 * η)
    num_steps = model.prior_sgld_steps
    prior_noise = st_rng.prior_noise

    state = (1, z_init)
    @trace while first(state) <= num_steps
        i, z = state
        z_new = prior_step(
            i,
            z,
            model.energy_net,
            ps.ebm,
            st.ebm,
            η,
            σ,
            sqrt_2η,
            σ_prior²,
            prior_noise,
        )
        state = (i + 1, z_new)
    end

    return last(state)
end

function posterior_step(
        i,
        z,
        x,
        model,
        ps,
        st,
        η,
        σ,
        sqrt_2η,
        σ_prior²,
        post_noise,
    )
    ∇z = first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(log_posterior_sum),
            z,
            Enzyme.Const(x),
            Enzyme.Const(model),
            Enzyme.Const(ps),
            Enzyme.Const(st),
        )
    )

    # Gaussian prior: ∇ log p_gaussian(z) = -z/σ_prior²
    ∇z_prior = -z ./ σ_prior²

    ε = post_noise[:, :, i]
    z_new = z .+ η .* (∇z .+ ∇z_prior) .+ σ .* sqrt_2η .* ε
    return z_new
end

# ULA for posterior: sample z ~ p(z|x) ∝ p(x|z)p(z)
function langevin_posterior(model::PangEBM, x, ps, st, st_rng)
    σ_prior² = model.prior_sigma^2
    z_init = st_rng.post_init
    η = model.post_sgld_step_size
    σ = model.noise_scale
    sqrt_2η = sqrt(2.0f0 * η)
    num_steps = model.post_sgld_steps
    post_noise = st_rng.post_noise

    state = (1, z_init)
    @trace while first(state) <= num_steps
        i, z = state
        z_new = posterior_step(
            i,
            z,
            x,
            model,
            ps,
            st,
            η,
            σ,
            sqrt_2η,
            σ_prior²,
            post_noise,
        )
        state = (i + 1, z_new)
    end

    return last(state)
end

function generate_pang(model::PangEBM, ps, st, st_rng)
    z = langevin_prior(model, ps, st, st_rng)
    x_gen, st_new = model(ps, st, z)
    return x_gen, st_new
end

end
