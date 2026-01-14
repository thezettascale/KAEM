module PangLoss

export PangTrainStep

using Enzyme, Optimisers, Lux, Statistics
using Flux: mse

using ..PangEBMSampling: langevin_prior, langevin_posterior

function pang_total_loss(ps, x, z_prior, z_post, model, st, α_cd)
    E_post, _ = model.energy_net(z_post, ps.ebm, st.ebm)
    E_prior, _ = model.energy_net(z_prior, ps.ebm, st.ebm)

    # Contrastive divergence: E[E(z_post)] - E[E(z_prior)]
    cd_loss = mean(E_post) - mean(E_prior)

    # Reconstruction loss from posterior samples
    x_recon, _ = model.generator(z_post, ps.gen, st.gen)
    recon_loss = mse(x_recon, x)
    loss = recon_loss + α_cd * cd_loss
    return (loss, st)
end

struct PangTrainStep
    model
    α_cd
end

function (l::PangTrainStep)(opt_state, ps, st, x, st_rng)
    z_prior = langevin_prior(l.model, ps, st, st_rng)
    z_post = langevin_posterior(l.model, x, ps, st, st_rng)

    dps = Enzyme.make_zero(ps)
    _, (loss, st_new) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        Const(pang_total_loss),
        Active,
        Duplicated(ps, dps),
        Const(x),
        Const(z_prior),
        Const(z_post),
        Const(l.model),
        Const(Lux.trainmode(st)),
        Const(l.α_cd),
    )

    opt_state, ps = Optimisers.update(opt_state, ps, dps)
    return loss, ps, opt_state, st_new
end

end
