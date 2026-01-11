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
        p_value;
        swap_replica_idxs = nothing,
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

    # Steppingstone estimator
    num_temps = model.N_t > 1 ? model.N_t : 1

    log_ss = 0.0f0
    for t in 1:num_temps

        noise_t = (
            model.lkhood.SEQ ? tempered_noise[:, :, :, t] : (
                    model.use_pca ? tempered_noise[:, :, t] :
                    tempered_noise[:, :, :, :, t]
                )
        )

        ll, st_gen = log_likelihood_MALA(
            z_posterior[:, :, :, t],
            x,
            model.lkhood,
            ps.gen,
            st_kan.gen,
            st_lux_gen,
            noise_t;
            ε = model.ε,
        )
        log_ss += Δt[t] * mean(ll)
    end

    # MLE estimator
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

    logllhood, st_gen = log_likelihood_MALA(
        z_prior,
        x,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_gen,
        noise;
        ε = model.ε,
    )
    steppingstone_loss = Δt[1] * mean(logllhood) + log_ss

    reg, st_ebm, st_gen = model.kan_regularizer(
        z_posterior[:, :, :, num_temps],
        model,
        ps,
        st_kan,
        st_ebm,
        st_gen
    )

    return reg - (steppingstone_loss + mean(logprior_pos) - ex_prior),
        st_ebm,
        st_gen
end

function closure(
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
    return first(
        marginal_llhood(
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
        ),
    )
end

function grad_thermo_llhood(
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
    return first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(closure),
            ps,
            Enzyme.Const(z_posterior),
            Enzyme.Const(z_prior),
            Enzyme.Const(x),
            Enzyme.Const(Δt),
            Enzyme.Const(model),
            Enzyme.Const(st_kan),
            Enzyme.Const(st_lux_ebm),
            Enzyme.Const(st_lux_gen),
            Enzyme.Const(noise),
            Enzyme.Const(tempered_noise),
            Enzyme.Const(component_mask),
        )
    )
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
        Lux.trainmode(st_lux),
        l.model,
        x,
        st_rng,
        compute_p(l.model, train_idx),
    )
    st_lux_ebm, st_lux_gen = st_lux.ebm, st_lux.gen
    z_prior, st_ebm =
        l.model.sample_prior(l.model, ps, st_kan, st_lux, st_rng)

    ∇ = grad_thermo_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        Δt,
        l.model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
        tempered_noise,
        component_mask,
    )

    loss, st_lux_ebm, st_lux_gen = marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        Δt,
        l.model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
        tempered_noise,
        component_mask,
    )

    opt_state, ps = Optimisers.update(opt_state, ps, ∇)
    return loss, ps, opt_state, st_lux_ebm, st_lux_gen
end

end
