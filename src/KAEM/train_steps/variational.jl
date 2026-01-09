module VariationalTraining

export VariationalLoss

using ComponentArrays, Enzyme, Statistics, Lux, Optimisers, NNlib

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
include("../ebm/mixture_selection.jl")
using .LogLikelihoods: log_likelihood_MALA
using .MixtureChoice: choose_component

function sample_encoder(
        ps,
        st_kan,
        st_lux,
        model,
        x,
        st_rng,
    )
    ε = st_rng.encoder_noise
    Q, P, S = model.prior.q_size, model.prior.p_size, model.batch_size

    # Get component mask for mixture model
    component_mask = (
        model.encoder.bool_config.mixture_model ?
            choose_component(ps.ebm.dist.α, S, Q, P, st_rng) :
            nothing
    )

    z, log_q, μ, logvar, st_enc = model.encoder(
        ps.enc,
        st_lux.enc,
        x,
        ε;
        component_mask = component_mask,
    )
    noise = st_rng.train_noise
    return z, log_q, st_enc, noise
end

function elbo_loss(
        ps,
        z_posterior,
        log_q,
        x,
        model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
        β
    )
    log_p, st_ebm =
        model.log_prior(
        z_posterior,
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm;
        ula = true
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

    kl = log_q .- log_p

    reg, st_ebm, st_gen = model.kan_regularizer(
        z_posterior,
        model,
        ps,
        st_kan,
        st_ebm,
        st_gen
    )

    return reg - (mean(logllhood) - β * mean(kl)),
        st_ebm,
        st_gen
end

function closure(
        ps,
        z_posterior,
        log_q,
        x,
        model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
        β
    )
    return first(
        elbo_loss(
            ps,
            z_posterior,
            log_q,
            x,
            model,
            st_kan,
            st_lux_ebm,
            st_lux_gen,
            noise,
            β,
        ),
    )
end

function grad_elbo(
        ps,
        z_posterior,
        log_q,
        x,
        model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
        β
    )
    return first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(closure),
            ps,
            Enzyme.Const(z_posterior),
            Enzyme.Const(log_q),
            Enzyme.Const(x),
            Enzyme.Const(model),
            Enzyme.Const(st_kan),
            Enzyme.Const(st_lux_ebm),
            Enzyme.Const(st_lux_gen),
            Enzyme.Const(noise),
            Enzyme.Const(β)
        )
    )
end

struct VariationalLoss
    model
    beta::AbstractArray{Float32}
end

function (l::VariationalLoss)(
        opt_state,
        ps,
        st_kan,
        st_lux,
        x,
        train_idx,
        st_rng,
    )
    β = l.beta[train_idx]

    z_posterior, log_q, st_enc, noise = sample_encoder(
        ps,
        st_kan,
        Lux.trainmode(st_lux),
        l.model,
        x,
        st_rng
    )
    st_lux_ebm, st_lux_gen = Lux.trainmode(st_lux.ebm), Lux.trainmode(st_lux.gen)

    ∇ = grad_elbo(
        ps,
        z_posterior,
        log_q,
        x,
        l.model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
        β,
    )

    loss, st_lux_ebm, st_lux_gen = elbo_loss(
        ps,
        z_posterior,
        log_q,
        x,
        l.model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
        β,
    )

    opt_state, ps = Optimisers.update(opt_state, ps, ∇)
    return loss, ps, opt_state, st_lux_ebm, st_lux_gen
end

end
