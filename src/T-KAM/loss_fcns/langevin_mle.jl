module LangevinMLE

export LangevinLoss, initialize_langevin_loss

using CUDA, KernelAbstractions, Enzyme, ComponentArrays, Random
using Statistics, Lux, LuxCUDA

using ..Utils
using ..T_KAM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

function sample_langevin(
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    model::T_KAM{T,full_quant},
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
    model::T_KAM{T,full_quant},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple;
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}

    logprior_pos, st_ebm =
        model.log_prior(z_posterior, model.prior, ps.ebm, st_kan.ebm, st_lux_ebm)
    logllhood, st_lux_gen = log_likelihood_MALA(
        z_posterior,
        x,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux_gen;
        ε = model.ε,
    )

    logprior, st_ebm = model.log_prior(z_prior, model.prior, ps.ebm, st_kan.ebm, st_lux_ebm)
    ex_prior = model.prior.contrastive_div ? mean(logprior) : zero(T)
    return -(mean(logprior_pos) + mean(logllhood) - ex_prior)*model.loss_scaling,
    st_lux_ebm,
    st_lux_gen
end

function closure(
    ps::ComponentArray{T},
    z_posterior::AbstractArray{T},
    z_prior::AbstractArray{T},
    x::AbstractArray{T},
    model::T_KAM{T,full_quant},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple;
)::T where {T<:half_quant}
    return first(
        marginal_llhood(ps, z_posterior, z_prior, x, model, st_kan, st_lux_ebm, st_lux_gen),
    )
end

function grad_langevin_llhood(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    z_posterior::AbstractArray{T},
    z_prior::AbstractArray{T},
    x::AbstractArray{T},
    model::T_KAM{T,full_quant},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple;
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    CUDA.@fastmath Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(closure),
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

struct LangevinLoss end

function (l::LangevinLoss)(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    model::T_KAM{T,full_quant},
    x::AbstractArray{T};
    train_idx::Int = 1,
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{T,AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    z_posterior, st_new =
        sample_langevin(ps, st_kan, Lux.testmode(st_lux), model, x; rng = rng)
    st_lux_ebm, st_lux_gen = st_new.ebm, st_new.gen
    z_prior, st_lux_ebm = model.sample_prior(model, size(x)[end], ps, st_kan, st_lux, rng)

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
        Lux.trainmode(st_lux_ebm),
        Lux.trainmode(st_lux_gen),
    )
    return loss, ∇, st_lux_ebm, st_lux_gen
end

end
