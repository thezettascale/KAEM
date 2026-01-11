module GANLoss

export GANTrainStep

using Enzyme, Optimisers, Lux, Statistics, Accessors
using Flux: logitbinarycrossentropy

using ..GANArchitecture: generate, discriminate

function discriminator_loss(ps_disc, x_real, x_fake, disc, st_disc)
    logits_real, st_disc_new = discriminate(
        disc, x_real, ps_disc, Lux.trainmode(st_disc)
    )
    logits_fake, st_disc_new = discriminate(disc, x_fake, ps_disc, st_disc_new)

    loss_real = logitbinarycrossentropy(logits_real, one(eltype(logits_real)))
    loss_fake = logitbinarycrossentropy(logits_fake, zero(eltype(logits_fake)))
    return loss_real + loss_fake, st_disc_new
end

function generator_loss(ps_gen, z, gen, disc, ps_disc, st_gen, st_disc)
    x_fake, st_gen_new = generate(gen, z, ps_gen, Lux.trainmode(st_gen))
    logits_fake, _ = discriminate(disc, x_fake, ps_disc, Lux.testmode(st_disc))
    loss = logitbinarycrossentropy(logits_fake, one(eltype(logits_fake)))
    return loss, st_gen_new, x_fake
end

function disc_closure(ps_disc, x_real, x_fake, disc, st_disc)
    return first(discriminator_loss(ps_disc, x_real, x_fake, disc, st_disc))
end

function gen_closure(ps_gen, z, gen, disc, ps_disc, st_gen, st_disc)
    return first(generator_loss(ps_gen, z, gen, disc, ps_disc, st_gen, st_disc))
end

function grad_disc(ps_disc, x_real, x_fake, disc, st_disc)
    return first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(disc_closure),
            ps_disc,
            Enzyme.Const(x_real),
            Enzyme.Const(x_fake),
            Enzyme.Const(disc),
            Enzyme.Const(st_disc),
        )
    )
end

function grad_gen(ps_gen, z, gen, disc, ps_disc, st_gen, st_disc)
    return first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(gen_closure),
            ps_gen,
            Enzyme.Const(z),
            Enzyme.Const(gen),
            Enzyme.Const(disc),
            Enzyme.Const(ps_disc),
            Enzyme.Const(st_gen),
            Enzyme.Const(st_disc),
        )
    )
end

struct GANTrainStep
    model
    n_critic
end

function (l::GANTrainStep)(opt_state_gen, opt_state_disc, ps, st, x_real, z, train_idx)
    gen = l.model.generator
    disc = l.model.discriminator
    x_fake, st_gen = generate(gen, z, ps.gen, Lux.testmode(st.gen))

    ∇_disc = grad_disc(ps.disc, x_real, x_fake, disc, st.disc)
    d_loss, st_disc_new = discriminator_loss(
        ps.disc, x_real, x_fake, disc, Lux.trainmode(st.disc)
    )
    opt_state_disc, ps_disc_new = Optimisers.update(opt_state_disc, ps.disc, ∇_disc)
    @reset ps.disc = ps_disc_new
    @reset st.disc = st_disc_new

    g_loss = 0.0f0
    ps_gen_new = ps.gen
    st_gen_new = st.gen
    if train_idx % l.n_critic == 0
        ∇_gen = grad_gen(ps_gen_new, z, gen, disc, ps.disc, st_gen_new, st_disc_new)
        g_loss, st_gen_new, _ = generator_loss(
            ps_gen_new, z, gen, disc, ps.disc, Lux.trainmode(st_gen_new), st_disc_new
        )
        opt_state_gen, ps_gen_new = Optimisers.update(opt_state_gen, ps_gen_new, ∇_gen)
    end
    @reset ps.gen = ps_gen_new
    @reset st.gen = st_gen_new

    return d_loss + g_loss, ps, opt_state_gen, opt_state_disc, st
end

end
