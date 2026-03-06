module ThermodynamicIntegration

export ThermoLoss

using ComponentArrays, Enzyme, Statistics, Lux, Optimisers

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
include("../ebm/mixture_selection.jl")
using .LogLikelihoods: log_likelihood_MALA
using .MixtureChoice: choose_component

function sample_thermo(
        ps,
        st_kan,
        st_lux,
        model,
        x,
        st_rng,
        p_value,
    )
    temps = collect(Float32, [(k / model.N_t)^p_value for k in 0:model.N_t])
    z, st_lux = model.posterior_sampler(
        ps,
        st_kan,
        st_lux,
        x,
        st_rng;
        temps = temps[2:end],
    )

    Δt = temps[2:end] - temps[1:(end - 1)]
    tempered_noise = st_rng.tempered_noise
    noise = st_rng.train_noise

    # Get component mask for mixture model normalization
    Q, P, S = model.prior.q_size, model.prior.p_size, model.batch_size
    component_mask = (
        model.prior.bool_config.mixture_model && !model.prior.bool_config.contrastive_div ?
            choose_component(ps.ebm.dist.α, S, Q, P, st_rng) :
            nothing
    )

    return z, Δt', st_lux, noise, tempered_noise, component_mask
end

function marginal_llhood(
        ps,
        z,
        x,
        Δt,
        model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
        tempered_noise,
        component_mask,
    )

    num_temps = model.N_t > 1 ? model.N_t : 1
    Q, P, S = model.posterior_sampler.Q, model.posterior_sampler.P, model.batch_size

    ll, st_gen = log_likelihood_MALA(
        reshape(z, Q, P, S * (num_temps + 1)),
        x,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux_gen,
        tempered_noise;
        ε = model.ε,
    )

    # Trapezoidal: ½ Σ_k Δt_k (E_{k-1} + E_k)
    E = dropdims(mean(reshape(ll, S, num_temps + 1); dims = 1); dims = 1)
    log_ss = 0.5f0 * sum(Δt .* (E[1:num_temps] .+ E[2:(num_temps + 1)]))

    logprior_pos, st_ebm = model.log_prior(
        z[:, :, :, num_temps + 1],
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm,
        st_kan.quad;
        component_mask = component_mask,
    )

    logprior, st_ebm = model.log_prior(
        z[:, :, :, 1],
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_ebm,
        st_kan.quad;
        component_mask = component_mask
    )
    ex_prior = model.prior.bool_config.contrastive_div ? mean(logprior) : 0.0f0

    reg, st_ebm, st_gen = model.kan_regularizer(
        z[:, :, :, num_temps + 1],
        model,
        ps,
        st_kan,
        st_ebm,
        st_gen
    )

    loss = reg - (log_ss + mean(logprior_pos) - ex_prior)
    return (loss, st_ebm, st_gen)
end

struct ThermoLoss
    model
end

function (l::ThermoLoss)(
        opt_state,
        ps,
        st_kan,
        st_lux,
        x,
        train_idx,
        st_rng,
    )
    z, Δt, st_lux, noise, tempered_noise, component_mask = sample_thermo(
        ps,
        st_kan,
        st_lux,
        l.model,
        x,
        st_rng,
        compute_p(l.model, train_idx),
    )
    st_lux_ebm, st_lux_gen = st_lux.ebm, st_lux.gen
    z_prior, st_ebm =
        l.model.sample_prior(l.model, ps, st_kan, st_lux, st_rng)

    Q, P, S = l.model.posterior_sampler.Q, l.model.posterior_sampler.P, l.model.batch_size
    z = cat(z_prior, z; dims = 4)

    x = l.model.lkhood.SEQ ? repeat(x, 1, 1, l.model.N_t + 1) :
        (l.model.use_pca ? repeat(x, 1, l.model.N_t + 1) : repeat(x, 1, 1, 1, l.model.N_t + 1))

    dps = Enzyme.make_zero(ps)
    _, (loss, st_lux_ebm, st_lux_gen) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        Const(marginal_llhood),
        Active,
        Duplicated(ps, dps),
        Const(z),
        Const(x),
        Const(Δt),
        Const(l.model),
        Const(st_kan),
        Const(st_lux_ebm),
        Const(st_lux_gen),
        Const(noise),
        Const(tempered_noise),
        Const(component_mask),
    )

    opt_state, ps = Optimisers.update(opt_state, ps, dps)
    return loss, ps, opt_state, st_lux_ebm, st_lux_gen
end

end
