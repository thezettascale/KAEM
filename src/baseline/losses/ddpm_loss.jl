export DDPMTrainStep

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
    x_noisy = q_sample(x_0, sqrt_alpha, sqrt_one_minus_alpha, noise)
    noise_pred, st_new = model(x_noisy, t, ps, Lux.trainmode(st))
    loss = Flux.mse(noise, noise_pred)
    return loss, st_new
end

function ddpm_closure(
        ps,
        x_0,
        t,
        sqrt_alpha,
        sqrt_one_minus_alpha,
        noise,
        model,
        st,
    )
    return first(
        noise_pred_loss(
            ps,
            x_0,
            t,
            sqrt_alpha,
            sqrt_one_minus_alpha,
            noise,
            model,
            st
        )
    )
end

function grad_ddpm(
        ps,
        x_0,
        t,
        sqrt_alpha,
        sqrt_one_minus_alpha,
        noise,
        model,
        st,
    )
    return first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(ddpm_closure),
            ps,
            Enzyme.Const(x_0),
            Enzyme.Const(t),
            Enzyme.Const(sqrt_alpha),
            Enzyme.Const(sqrt_one_minus_alpha),
            Enzyme.Const(noise),
            Enzyme.Const(model),
            Enzyme.Const(st),
        )
    )
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
    ∇ = grad_ddpm(
        ps,
        x_0,
        t,
        sqrt_alpha,
        sqrt_one_minus_alpha,
        noise,
        l.model,
        st
    )
    loss, st_new = noise_pred_loss(
        ps,
        x_0,
        t,
        sqrt_alpha,
        sqrt_one_minus_alpha,
        noise,
        l.model,
        Lux.trainmode(st)
    )
    opt_state, ps = Optimisers.update(opt_state, ps, ∇)
    return loss, ps, opt_state, st_new
end
