module GANLoss

export GANTrainStep

using Enzyme, Optimisers, Lux, Statistics, Accessors, ComponentArrays
using Flux: logitbinarycrossentropy

using ..Utils

function discriminator_loss(ps_disc, x_real, x_fake, disc, st_disc)
    logits_real, st_disc_new = disc(x_real, ps_disc, Lux.trainmode(st_disc))
    logits_fake, st_disc_new = disc(x_fake, ps_disc, st_disc_new)

    real_labels = zero(logits_real) .+ 1
    fake_labels = zero(logits_fake)

    loss_real = logitbinarycrossentropy(logits_real, real_labels; agg = mean)
    loss_fake = logitbinarycrossentropy(logits_fake, fake_labels; agg = mean)
    return (loss_real + loss_fake, st_disc_new)
end

function generator_loss(ps_gen, z, gen, disc, ps_disc, st_gen, st_disc)
    x_fake, st_gen_new = gen(z, ps_gen, Lux.trainmode(st_gen))
    logits_fake, _ = disc(x_fake, ps_disc, Lux.testmode(st_disc))

    real_labels = zero(logits_fake) .+ 1
    loss = logitbinarycrossentropy(logits_fake, real_labels; agg = mean)
    return (loss, st_gen_new, x_fake)
end

struct GANTrainStep
    model
    n_critic
end

function (l::GANTrainStep)(opt_state_gen, opt_state_disc, ps, st, x_real, st_rng, train_idx)
    gen = l.model.generator
    disc = l.model.discriminator
    z = st_rng.z

    x_fake, st_gen = gen(z, ps.gen, Lux.trainmode(st.gen))

    # Discriminator update
    dps_disc = Enzyme.make_zero(ps.disc)
    _, (d_loss, st_disc_new) = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        Const(discriminator_loss),
        Active,
        Duplicated(ps.disc, dps_disc),
        Const(x_real),
        Const(x_fake),
        Const(disc),
        Const(st.disc),
    )
    opt_state_disc, ps_disc_new = Optimisers.update(opt_state_disc, ps.disc, dps_disc)
    @views ps[:disc] .= ps_disc_new
    @reset st.disc = st_disc_new

    # Generator update (conditional on n_critic)
    g_loss = 0.0f0
    ps_gen_new = ps.gen
    st_gen_new = st_gen
    if train_idx % l.n_critic == 0
        dps_gen = Enzyme.make_zero(ps_gen_new)
        _, (g_loss, st_gen_new, _) = Enzyme.autodiff(
            Enzyme.ReverseWithPrimal,
            Const(generator_loss),
            Active,
            Duplicated(ps_gen_new, dps_gen),
            Const(z),
            Const(gen),
            Const(disc),
            Const(ps.disc),
            Const(st_gen_new),
            Const(st_disc_new),
        )
        opt_state_gen, ps_gen_new = Optimisers.update(opt_state_gen, ps_gen_new, dps_gen)
    end
    @views ps[:gen] .= ps_gen_new
    @reset st.gen = st_gen_new

    return d_loss + g_loss, ps, opt_state_gen, opt_state_disc, st
end

end
