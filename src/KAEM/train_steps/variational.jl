module VariationalTraining

export VariationalLoss

using ComponentArrays, Enzyme, Statistics, Lux, Optimisers, NNlib

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
include("../ebm/mixture_selection.jl")
using .LogLikelihoods: log_likelihood_MALA
using .MixtureChoice: choose_component

function elbo_loss(
        ps,
        x,
        ε,
        model,
        st_kan,
        st_lux_enc,
        st_lux_ebm,
        st_lux_gen,
        noise,
        β,
        component_mask,
    )
    z_posterior, log_q, μ, logvar, st_enc = model.encoder(
        ps.enc,
        st_lux_enc,
        x,
        ε;
        component_mask = component_mask,
    )

    log_p, st_ebm = model.log_prior(
        z_posterior,
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm,
        st_kan.quad;
        ula = false,
        component_mask = component_mask,
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
        st_gen,
        st_enc
end

function closure(
        ps,
        x,
        ε,
        model,
        st_kan,
        st_lux_enc,
        st_lux_ebm,
        st_lux_gen,
        noise,
        β,
        component_mask,
    )
    return first(
        elbo_loss(
            ps,
            x,
            ε,
            model,
            st_kan,
            st_lux_enc,
            st_lux_ebm,
            st_lux_gen,
            noise,
            β,
            component_mask,
        ),
    )
end

function grad_elbo(
        ps,
        x,
        ε,
        model,
        st_kan,
        st_lux_enc,
        st_lux_ebm,
        st_lux_gen,
        noise,
        β,
        component_mask,
    )
    return first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(closure),
            ps,
            Enzyme.Const(x),
            Enzyme.Const(ε),
            Enzyme.Const(model),
            Enzyme.Const(st_kan),
            Enzyme.Const(st_lux_enc),
            Enzyme.Const(st_lux_ebm),
            Enzyme.Const(st_lux_gen),
            Enzyme.Const(noise),
            Enzyme.Const(β),
            Enzyme.Const(component_mask),
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

    ε = st_rng.encoder_noise
    noise = st_rng.train_noise
    Q, P, S = l.model.prior.q_size, l.model.prior.p_size, l.model.batch_size

    component_mask = (
        l.model.encoder.bool_config.mixture_model ?
            choose_component(ps.ebm.dist.α, S, Q, P, st_rng) :
            nothing
    )

    st_lux_enc = Lux.trainmode(st_lux.enc)
    st_lux_ebm = Lux.trainmode(st_lux.ebm)
    st_lux_gen = Lux.trainmode(st_lux.gen)

    ∇ = grad_elbo(
        ps,
        x,
        ε,
        l.model,
        st_kan,
        st_lux_enc,
        st_lux_ebm,
        st_lux_gen,
        noise,
        β,
        component_mask,
    )

    loss, st_lux_ebm, st_lux_gen, st_lux_enc = elbo_loss(
        ps,
        x,
        ε,
        l.model,
        st_kan,
        st_lux_enc,
        st_lux_ebm,
        st_lux_gen,
        noise,
        β,
        component_mask,
    )

    opt_state, ps = Optimisers.update(opt_state, ps, ∇)
    return loss, ps, opt_state, st_lux_ebm, st_lux_gen
end

end
