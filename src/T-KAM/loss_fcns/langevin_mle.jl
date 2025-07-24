module LangevinMLE

export LangevinLoss

using CUDA, KernelAbstractions, Enzyme, ComponentArrays, Random
using Statistics, Lux, LuxCUDA

include("../gen/loglikelihoods.jl")
include("../../utils.jl")
using .LogLikelihoods: log_likelihood_MALA
using .Utils: device, half_quant, full_quant, hq

function sample_langevin(
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    model,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
    z, st_lux, = model.posterior_sampler(model, ps, st_kan, st_lux, x; rng = rng)
    return z[:, :, :, 1], st_lux
end

function marginal_llhood(
    ps::ComponentArray{T},
    z_posterior::AbstractArray{T},
    z_prior::AbstractArray{T},
    x::AbstractArray{T},
    model,
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple;
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}

    logprior_pos, st_ebm = model.prior.lp_fcn(
        z_posterior,
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm;
        ε = model.ε,
        normalize = !model.prior.contrastive_div,
    )
    logllhood, st_lux_gen = log_likelihood_MALA(
        z_posterior,
        x,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux_gen;
        ε = model.ε,
    )

    logprior, st_ebm = model.prior.lp_fcn(
        z_prior,
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm;
        ε = model.ε,
        normalize = !model.prior.contrastive_div,
    )
    ex_prior = model.prior.contrastive_div ? mean(logprior) : zero(T)
    return -(mean(logprior_pos) + mean(logllhood) - ex_prior)*model.loss_scaling,
    st_lux_ebm,
    st_lux_gen
end

function grad_langevin_llhood(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    z_posterior::AbstractArray{T},
    z_prior::AbstractArray{T},
    x::AbstractArray{T},
    model,
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple;
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    f =
        (p, post_i, prior_i, x_i, m, sk, se, sg) -> begin
            first(marginal_llhood(p, post_i, prior_i, x_i, m, sk, se, sg))
        end

    CUDA.@fastmath Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        f,
        Enzyme.Active,
        Enzyme.Duplicated(ps, ∇),
        Enzyme.Const(z_posterior),
        Enzyme.Const(z_prior),
        Enzyme.Const(x),
        Enzyme.Const(model),
        Enzyme.Const(st_kan),
        Enzyme.Const(st_lux_ebm),
        Enzyme.Const(st_lux_gen),
    )

    return ∇, st_lux_ebm, st_lux_gen
end

struct LangevinLoss
end

function (l::LangevinLoss)(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    model,
    x::AbstractArray{T};
    train_idx::Int = 1,
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{T,AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    z_posterior, st_new =
        sample_langevin(ps, st_kan, Lux.testmode(st_lux), model, x; rng = rng)
    st_lux_ebm, st_lux_gen = st_new.ebm, st_new.gen
    z_prior, st_lux_ebm = model.prior.sample_z(model, size(x)[end], ps, st_kan, st_lux, rng)

    ∇, st_lux_ebm, st_lux_gen = grad_langevin_llhood(
        ps,
        ∇,
        z_posterior,
        z_prior,
        x,
        model,
        st_kan,
        Lux.trainmode(st_lux_ebm),
        Lux.trainmode(st_lux_gen),
    )
    loss, st_lux_ebm, st_lux_gen = marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        model,
        st_kan,
        Lux.testmode(st_lux_ebm),
        Lux.testmode(st_lux_gen),
    )
    return loss, ∇, st_lux_ebm, st_lux_gen
end

end
