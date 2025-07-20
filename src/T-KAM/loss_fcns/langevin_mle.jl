module LangevinMLE

export initialize_langevin_loss, langevin_loss

using CUDA, KernelAbstractions, Enzyme, ComponentArrays, Random, Reactant
using Statistics, Lux, LuxCUDA

include("../gen/loglikelihoods.jl")
include("../../utils.jl")
using .LogLikelihoods: log_likelihood_MALA
using .Utils: device, half_quant, full_quant, hq

function sample_langevin(
    ps::ComponentArray{T},
    st::NamedTuple,
    m,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
    z, st_ebm = m.posterior_sample(m, x, ps, st, rng)
    return z[:, :, :, 1], st_ebm
end

function marginal_llhood(
    ps::ComponentArray{T},
    z_posterior::AbstractArray{T},
    z_prior::AbstractArray{T},
    x::AbstractArray{T},
    m,
    st_ebm::NamedTuple,
    st_gen::NamedTuple;
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}

    logprior_pos, st_ebm = m.prior.lp_fcn(
        z_posterior,
        m.prior,
        ps.ebm,
        st_ebm;
        ε = m.ε,
        normalize = !m.prior.contrastive_div,
    )
    logllhood, st_gen =
        log_likelihood_MALA(z_posterior, x, m.lkhood, ps.gen, st_gen; ε = m.ε)

    logprior, st_ebm = m.prior.lp_fcn(
        z_prior,
        m.prior,
        ps.ebm,
        st_ebm;
        ε = m.ε,
        normalize = !m.prior.contrastive_div,
    )
    ex_prior = m.prior.contrastive_div ? mean(logprior) : zero(T)
    return -(mean(logprior_pos) + mean(logllhood) - ex_prior)*m.loss_scaling, st_ebm, st_gen
end

function grad_langevin_llhood(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    z_posterior::AbstractArray{T},
    z_prior::AbstractArray{T},
    x::AbstractArray{T},
    model,
    st_ebm::NamedTuple,
    st_gen::NamedTuple;
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    f =
        (p, post_i, prior_i, x_i, m, se, sg) -> begin
            first(marginal_llhood(p, post_i, prior_i, x_i, m, se, sg))
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
        Enzyme.Const(st_ebm),
        Enzyme.Const(st_gen),
    )

    return ∇, st_ebm, st_gen
end

struct LangevinLoss
    compiled_loss::Any
    compiled_grad::Any
end

function initialize_langevin_loss(
    ps::ComponentArray{T},
    st::NamedTuple,
    model,
    x::AbstractArray{T};
    compile_mlir::Bool = false,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:half_quant}
    ∇ = Enzyme.make_zero(ps)
    z_posterior, st_new = sample_langevin(ps, Lux.testmode(st), model, x; rng = rng)
    st_ebm, st_gen = st_new.ebm, st_new.gen
    z_prior, st_ebm = model.prior.sample_z(model, size(x)[end], ps, Lux.testmode(st), rng)
    compiled_loss = marginal_llhood
    compiled_grad = grad_langevin_llhood

    if compile_mlir
        compiled_loss = Reactant.@compile marginal_llhood(
            ps,
            z_posterior,
            z_prior,
            x,
            model,
            Lux.testmode(st_ebm),
            Lux.testmode(st_gen),
        )
        compiled_grad = Reactant.@compile grad_langevin_llhood(
            ps,
            ∇,
            z_posterior,
            z_prior,
            x,
            model,
            Lux.trainmode(st_ebm),
            Lux.trainmode(st_gen),
        )
    end

    return LangevinLoss(compiled_loss, compiled_grad)
end

function langevin_loss(
    l,
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    st::NamedTuple,
    model,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{T,AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    z_posterior, st_new = sample_langevin(ps, Lux.testmode(st), model, x; rng = rng)
    st_ebm, st_gen = st_new.ebm, st_new.gen
    z_prior, st_ebm = model.prior.sample_z(model, size(x)[end], ps, Lux.testmode(st), rng)

    ∇, st_ebm, st_gen = l.compiled_grad(
        ps,
        ∇,
        z_posterior,
        z_prior,
        x,
        model,
        Lux.trainmode(st_ebm),
        Lux.trainmode(st_gen),
    )
    loss, st_ebm, st_gen = l.compiled_loss(
        ps,
        z_posterior,
        z_prior,
        x,
        model,
        Lux.testmode(st_ebm),
        Lux.testmode(st_gen),
    )
    return loss, ∇, st_ebm, st_gen
end

end
