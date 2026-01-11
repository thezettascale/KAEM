module VAELoss

export VAETrainStep

using Enzyme, Optimisers, Lux, Statistics
using Flux: mse

function elbo_loss(ps, x, ε, model, st)
    x_recon, μ, logvar, st_new = model(ps, st, x, ε)
    recon_loss = mse(x_recon, x)

    # KL divergence (mean instead of sum for scale-invariance to latent dim)
    kl_loss = -0.5f0 * mean(1.0f0 .+ logvar .- μ .^ 2 .- exp.(logvar))
    return recon_loss + kl_loss, st_new, recon_loss, kl_loss
end

function vae_closure(ps, x, ε, model, st)
    return first(elbo_loss(ps, x, ε, model, st))
end

function grad_elbo(ps, x, ε, model, st)
    return first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(vae_closure),
            ps,
            Enzyme.Const(x),
            Enzyme.Const(ε),
            Enzyme.Const(model),
            Enzyme.Const(st),
        )
    )
end

struct VAETrainStep
    model
    β
end

function (l::VAETrainStep)(opt_state, ps, st, x, ε)
    ∇ = grad_elbo(ps, x, ε, l.model, Lux.trainmode(st))
    loss, st_new, recon_loss, kl_loss = elbo_loss(
        ps, x, ε, l.model, Lux.trainmode(st)
    )
    opt_state, ps = Optimisers.update(opt_state, ps, ∇)
    return loss, ps, opt_state, st_new
end

end
