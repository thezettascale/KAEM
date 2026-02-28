module ThermodynamicIntegration

export ThermoLoss

using ComponentArrays, Enzyme, Statistics, Lux, Optimisers
using NNlib: logsumexp

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

    return z, Δt, st_lux, noise, tempered_noise, component_mask
end

function marginal_llhood(
        ps,
        z_posterior,
        z_prior,
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

    # SS estimator = sum_{k=1}^{N_t} [ logsumexp(Δt_k * ll(z_{t_{k-1}})) - log(N) ]
    num_temps = model.N_t > 1 ? model.N_t : 1

    # k=1: samples from t_0 = 0 (prior)
    ll, st_gen = log_likelihood_MALA(
        z_prior,
        x,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux_gen,
        noise;
        ε = model.ε,
    )
    log_ss = logsumexp(Δt[1] .* ll) - log(model.batch_size)

    # k=2,...,N_t: samples from t_{k-1} (previous power posterior)
    for k in 2:num_temps

        noise_t = (
            model.lkhood.SEQ ? tempered_noise[:, :, :, k - 1] : (
                    model.use_pca ? tempered_noise[:, :, k - 1] :
                    tempered_noise[:, :, :, :, k - 1]
                )
        )

        ll, st_gen = log_likelihood_MALA(
            z_posterior[:, :, :, k - 1],
            x,
            model.lkhood,
            ps.gen,
            st_kan.gen,
            st_gen,
            noise_t;
            ε = model.ε,
        )
        log_ss += logsumexp(Δt[k] .* ll) - log(model.batch_size)
    end

    # MLE estimator (prior learned from full posterior samples at t_{N_t}=1)
    logprior_pos, st_ebm = model.log_prior(
        z_posterior[:, :, :, num_temps],
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm,
        st_kan.quad;
        component_mask = component_mask,
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
        z_posterior[:, :, :, num_temps],
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
    z_posterior, Δt, st_lux, noise, tempered_noise, component_mask = sample_thermo(
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

    dps = Enzyme.make_zero(ps)
    _, (loss, st_lux_ebm, st_lux_gen) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        Const(marginal_llhood),
        Active,
        Duplicated(ps, dps),
        Const(z_posterior),
        Const(z_prior),
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
