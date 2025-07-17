module LangevinMLE

export langevin_loss

using CUDA, KernelAbstractions, Enzyme, ComponentArrays
using Statistics, Lux, LuxCUDA

include("../../utils.jl")
using .Utils: device, half_quant, full_quant, hq

function sample_langevin(
    ps::ComponentArray{T},
    st::NamedTuple,
    m::Any,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
    z, st_ebm = m.posterior_sample(m, x, 0, ps, st, rng)
    return z, st_ebm
end

function marginal_llhood(
    ps::ComponentArray{T},
    z::AbstractArray{T},
    x::AbstractArray{T},
    m::Any,
    st_ebm::NamedTuple,
    st_gen::NamedTuple;
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}

    logprior_pos, st_ebm = m.prior.lp_fcn(
        z[:, :, :, 1],
        m.prior,
        ps.ebm,
        st_ebm;
        ε = m.ε,
        normalize = !m.prior.contrastive_div,
    )
    logllhood, st_gen = log_likelihood_MALA(
        z[:, :, :, 1],
        x,
        m.lkhood,
        ps.gen,
        st_gen;
        rng = rng,
        ε = m.ε,
    )
    contrastive_div = mean(logprior_pos)

    if m.prior.contrastive_div
        z, st_ebm = m.prior.sample_z(m, size(x)[end], ps, st, rng)
        logprior, st_ebm = m.prior.lp_fcn(
            z,
            m.prior,
            ps.ebm,
            st_ebm;
            ε = m.ε,
            normalize = !m.prior.contrastive_div,
        )
        contrastive_div -= mean(logprior)
    end

    return -(contrastive_div + mean(logllhood))*m.loss_scaling, st_ebm, st_gen
end

function langevin_loss(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    st::NamedTuple,
    model::Any,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    """MLE loss without importance, (used when posterior expectation = MCMC estimate)."""

    z, st_new = sample_langevin(ps, st, model
    st_ebm, st_gen = st_new.ebm, st_new.gen

    f =
        (p, z_i, x_i, m, se, sg) -> begin
            first(marginal_llhood(p, z_i, x_i, m, se, sg; rng = rng))
        end

    CUDA.@fastmath Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        f,
        Enzyme.Active,
        Enzyme.Duplicated(ps, ∇),
        Enzyme.Const(z),
        Enzyme.Const(x),
        Enzyme.Const(model),
        Enzyme.Const(st_ebm),
        Enzyme.Const(st_gen),
    )

    loss, st_ebm, st_gen =
        marginal_llhood(ps, z, x, model, st_ebm, st_gen; rng = rng)
    return loss, ∇, st_ebm, st_gen
end

end
