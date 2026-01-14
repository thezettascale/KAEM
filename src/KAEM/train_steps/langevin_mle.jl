module LangevinMLE

export LangevinLoss

using ComponentArrays, Enzyme, Statistics, Lux, Optimisers

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
include("../ebm/mixture_selection.jl")
using .LogLikelihoods: log_likelihood_MALA
using .MixtureChoice: choose_component

function sample_langevin(
        ps,
        st_kan,
        st_lux,
        model,
        x,
        st_rng,
    )
    z, st_lux, = model.posterior_sampler(ps, st_kan, st_lux, x, st_rng)
    noise = st_rng.train_noise

    # Get component mask for mixture model normalization
    Q, P, S = model.prior.q_size, model.prior.p_size, model.batch_size
    component_mask = (
        model.prior.bool_config.mixture_model && !model.prior.bool_config.contrastive_div ?
            choose_component(ps.ebm.dist.α, S, Q, P, st_rng) :
            nothing
    )

    return z[:, :, :, 1], st_lux, noise, component_mask
end

function marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
        component_mask,
    )

    logprior_pos, st_ebm = model.log_prior(
        z_posterior,
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm,
        st_kan.quad;
        component_mask = component_mask
    )
    logllhood, st_gen = log_likelihood_MALA(
        z_posterior,
        x,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux_gen,
        noise;
        ε = model.ε,
    )

    logprior, st_ebm = model.log_prior(
        z_prior,
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_ebm,
        st_kan.quad;
        component_mask = component_mask
    )
    ex_prior = model.prior.bool_config.contrastive_div ? mean(logprior) : 0.0f0

    reg, st_ebm, st_gen = model.kan_regularizer(
        z_posterior,
        model,
        ps,
        st_kan,
        st_ebm,
        st_gen
    )

    loss = reg - (mean(logprior_pos) + mean(logllhood) - ex_prior)
    return (loss, st_ebm, st_gen)
end

struct LangevinLoss
    model
end

function (l::LangevinLoss)(
        opt_state,
        ps,
        st_kan,
        st_lux,
        x,
        train_idx,
        st_rng,
    )
    z_posterior, st_new, noise, component_mask =
        sample_langevin(ps, st_kan, Lux.trainmode(st_lux), l.model, x, st_rng)
    st_lux_ebm, st_lux_gen = st_new.ebm, st_new.gen
    z_prior, st_lux_ebm =
        l.model.sample_prior(l.model, ps, st_kan, st_lux, st_rng)

    dps = Enzyme.make_zero(ps)
    _, (loss, st_lux_ebm, st_lux_gen) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        Const(marginal_llhood),
        Active,
        Duplicated(ps, dps),
        Const(z_posterior),
        Const(z_prior),
        Const(x),
        Const(l.model),
        Const(st_kan),
        Const(st_lux_ebm),
        Const(st_lux_gen),
        Const(noise),
        Const(component_mask),
    )

    opt_state, ps = Optimisers.update(opt_state, ps, dps)
    return loss, ps, opt_state, st_lux_ebm, st_lux_gen
end

end
