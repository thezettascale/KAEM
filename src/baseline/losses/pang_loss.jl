module PangLoss

export PangTrainStep

using Enzyme, Optimisers, Lux, Statistics
using Flux: mse

using ..PangEBMSampling: langevin_prior, langevin_posterior

function pang_loss(ps, x, z_prior, z_post, model, st)
    E_post, _ = model.energy_net(z_post, ps.ebm, st.ebm)
    E_prior, _ = model.energy_net(z_prior, ps.ebm, st.ebm)

    # Contrastive divergence: E[E(z_post)] - E[E(z_prior)]
    cd_loss = mean(E_post) - mean(E_prior)

    # Reconstruction loss from posterior samples
    x_recon, _ = model.generator(z_post, ps.gen, st.gen)
    recon_loss = mse(x_recon, x)
    return recon_loss, cd_loss
end

function pang_total_loss(ps, x, z_prior, z_post, model, st, α_cd)
    recon, cd = pang_loss(ps, x, z_prior, z_post, model, st)
    return recon + α_cd * cd
end

function pang_closure(ps, x, z_prior, z_post, model, st, α_cd)
    return pang_total_loss(ps, x, z_prior, z_post, model, st, α_cd)
end

function grad_pang(ps, x, z_prior, z_post, model, st, α_cd)
    return first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(pang_closure),
            ps,
            Enzyme.Const(x),
            Enzyme.Const(z_prior),
            Enzyme.Const(z_post),
            Enzyme.Const(model),
            Enzyme.Const(st),
            Enzyme.Const(α_cd),
        )
    )
end

struct PangTrainStep
    model
    α_cd
end

function (l::PangTrainStep)(opt_state, ps, st, x, st_rng)
    z_prior = langevin_prior(l.model, ps, st, st_rng)
    z_post = langevin_posterior(l.model, x, ps, st, st_rng)

    ∇ = grad_pang(ps, x, z_prior, z_post, l.model, Lux.trainmode(st), l.α_cd)

    recon, cd = pang_loss(ps, x, z_prior, z_post, l.model, Lux.trainmode(st))
    loss = recon + l.α_cd * cd

    opt_state, ps = Optimisers.update(opt_state, ps, ∇)
    return loss, ps, opt_state, st
end

end
