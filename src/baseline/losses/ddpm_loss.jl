module DDPMLoss

export DDPMTrainStep

using Enzyme, Optimisers, Lux, Statistics
using Flux: mse

using ..Utils

function noise_pred_loss(
        ps,
        x_0,
        t,
        sqrt_alpha,
        sqrt_one_minus_alpha,
        noise,
        model,
        st,
    )
    x_noisy = sqrt_alpha .* x_0 .+ sqrt_one_minus_alpha .* noise
    noise_pred, st_new = model(x_noisy, t, ps, Lux.trainmode(st))
    loss = mse(noise, noise_pred)
    return (loss, st_new)
end

struct DDPMTrainStep
    model
end

function (l::DDPMTrainStep)(
        opt_state,
        ps,
        st,
        x_0,
        t,
        sqrt_alpha,
        sqrt_one_minus_alpha,
        noise,
    )
    dps = Enzyme.make_zero(ps)

    (loss, st_new), _ = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        Const(noise_pred_loss),
        Active,
        Duplicated(ps, dps),
        Const(x_0),
        Const(t),
        Const(sqrt_alpha),
        Const(sqrt_one_minus_alpha),
        Const(noise),
        Const(l.model),
        Const(st),
    )

    opt_state, ps = Optimisers.update(opt_state, ps, dps)
    return loss, ps, opt_state, st_new
end

end
