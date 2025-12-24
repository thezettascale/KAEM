module ImportanceSampling

using ComponentArrays, Enzyme, Statistics, Lux, Optimisers
using NNlib: softmax

export ImportanceLoss

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_IS

function loss_accum(
        logprior,
        logllhood,
        resampled_mask,
        B,
        N,
    )
    sel_prior = sum(reshape(logprior, 1, 1, N) .* resampled_mask; dims = 3)
    is_prior = mean(sel_prior; dims = 2)

    ll = PermutedDimsArray(view(logllhood, :, :, :), (1, 3, 2))
    sel_ll = sum(ll .* resampled_mask; dims = 3)
    is_ll = mean(sel_ll; dims = 2)

    return mean(is_prior + is_ll)
end

function sample_importance(
        ps,
        st_kan,
        st_lux,
        m,
        x,
        st_rng
    )
    # Prior is proposal for importance sampling
    z_posterior, st_lux_ebm = m.sample_prior(m, ps, st_kan, st_lux, st_rng)
    noise = st_rng.train_noise
    logllhood, st_lux_gen = log_likelihood_IS(
        z_posterior,
        x,
        m.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux.gen,
        noise;
        ε = m.ε,
    )

    # Posterior weights and resampling
    N = m.batch_size
    weights = softmax(logllhood, dims = 2)
    resampled_indices = m.lkhood.resample_z(weights, st_rng)
    resampled_mask = resampled_indices .== reshape(1:N, 1, 1, N) |> Lux.f32

    # Works better with more samples
    z_prior, st_lux_ebm = m.sample_prior(m, ps, st_kan, st_lux, st_rng)
    return z_posterior,
        z_prior,
        st_lux_ebm,
        st_lux_gen,
        resampled_mask,
        noise
end

function marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        resampled_mask,
        m,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
    )
    B, N = m.batch_size, m.batch_size

    logprior_posterior, st_ebm =
        m.log_prior(z_posterior, m.prior, ps.ebm, st_kan.ebm, st_lux_ebm)
    logllhood, st_gen = log_likelihood_IS(
        z_posterior,
        x,
        m.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux_gen,
        noise;
        ε = m.ε,
    )

    logprior_prior, st_ebm =
        m.log_prior(z_prior, m.prior, ps.ebm, st_kan.ebm, st_ebm)
    ex_prior = m.prior.bool_config.contrastive_div ? mean(logprior_prior) : 0.0f0

    marginal_llhood = loss_accum(
        logprior_posterior,
        logllhood,
        resampled_mask,
        B,
        N
    )

    reg, st_ebm, st_gen = m.kan_regularizer(
        z_posterior,
        m,
        ps,
        st_kan,
        st_ebm,
        st_gen
    )

    return reg - (marginal_llhood - ex_prior), st_ebm, st_gen
end

function closure(
        ps,
        z_posterior,
        z_prior,
        x,
        resampled_mask,
        m,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
    )
    return first(
        marginal_llhood(
            ps,
            z_posterior,
            z_prior,
            x,
            resampled_mask,
            m,
            st_kan,
            st_lux_ebm,
            st_lux_gen,
            noise,
        ),
    )
end

function grad_importance_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        resampled_mask,
        model,
        st_kan,
        st_lux_ebm,
        st_lux_gen,
        noise,
    )

    return first(
        Enzyme.gradient(
            Enzyme.Reverse,
            Enzyme.Const(closure),
            ps,
            Enzyme.Const(z_posterior),
            Enzyme.Const(z_prior),
            Enzyme.Const(x),
            Enzyme.Const(resampled_mask),
            Enzyme.Const(model),
            Enzyme.Const(st_kan),
            Enzyme.Const(st_lux_ebm),
            Enzyme.Const(st_lux_gen),
            Enzyme.Const(noise),
        )
    )
end

struct ImportanceLoss
    model
end

function (l::ImportanceLoss)(
        opt_state,
        ps,
        st_kan,
        st_lux,
        x,
        train_idx,
        st_rng,
    )

    z_posterior, z_prior, st_lux_ebm, st_lux_gen, resampled_mask, noise =
        sample_importance(ps, st_kan, st_lux, l.model, x, st_rng)

    st_ebm = Lux.trainmode(st_lux_ebm)
    st_gen = Lux.trainmode(st_lux_gen)

    ∇ = grad_importance_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        resampled_mask,
        l.model,
        st_kan,
        st_ebm,
        st_gen,
        noise,
    )

    loss, st_lux_ebm, st_lux_gen = marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        resampled_mask,
        l.model,
        st_kan,
        st_ebm,
        st_gen,
        noise,
    )

    opt_state, ps = Optimisers.update(opt_state, ps, ∇)
    return loss, ps, opt_state, st_lux_ebm, st_lux_gen
end

end
