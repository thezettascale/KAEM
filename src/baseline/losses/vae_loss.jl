module VAELoss

export VAETrainStep

using Enzyme, Optimisers, Lux, Statistics
using Flux: mse

function elbo_loss(ps, x, ε, model, st)
    x_recon, μ, logvar, st_new = model(ps, st, x, ε)
    recon_loss = mse(x_recon, x)

    # KL divergence (mean instead of sum for scale-invariance to latent dim)
    kl_loss = -0.5f0 * mean(1.0f0 .+ logvar .- μ .^ 2 .- exp.(logvar))
    loss = recon_loss + kl_loss
    return (loss, st_new)
end

struct VAETrainStep
    model
    β
end

function (l::VAETrainStep)(opt_state, ps, st, x, ε)
    dps = Enzyme.make_zero(ps)

    (loss, st_new), _ = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        Const(elbo_loss),
        Active,
        Duplicated(ps, dps),
        Const(x),
        Const(ε),
        Const(l.model),
        Const(Lux.trainmode(st)),
    )

    opt_state, ps = Optimisers.update(opt_state, ps, dps)
    return loss, ps, opt_state, st_new
end

end
