module ImportanceSampling

using CUDA, KernelAbstractions, Enzyme, ComponentArrays
using Statistics, Lux, LuxCUDA
using NNlib: softmax

export importance_loss

include("../../utils.jl")
using .Utils: device, half_quant, full_quant, hq

function sample_importance(
    ps::ComponentArray{T},
    st::NamedTuple,
    m::Any,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{
    AbstractArray{T},
    NamedTuple,
    NamedTuple,
    AbstractArray{Int},
} where {T<:half_quant}
    z, st_ebm = m.prior.sample_z(m, m.IS_samples, ps, st, rng)
    logllhood, st_gen =
        log_likelihood_IS(z, x, m.lkhood, ps.gen, st.gen; rng = rng, ε = m.ε)
    weights = softmax(full_quant.(logllhood), dims = 2)
    resampled_idxs = m.lkhood.resample_z(weights, rng)
    weights = T.(weights)
    weights_resampled = softmax(
        reduce(vcat, map(b -> weights[b:b, resampled_idxs[b, :]], 1:size(x)[end])),
        dims = 2,
    )
    return z, st_ebm, st_gen, weights_resampled, resampled_idxs
end

function marginal_llhood(
    ps::ComponentArray{T},
    z::AbstractArray{T},
    x::AbstractArray{T},
    weights_resampled::AbstractArray{T},
    resampled_idxs::AbstractArray{Int},
    m::Any,
    st_ebm::NamedTuple,
    st_gen::NamedTuple;
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    logprior, st_ebm = m.prior.lp_fcn(
        z,
        m.prior,
        ps.ebm,
        st_ebm;
        ε = m.ε,
        normalize = !m.prior.contrastive_div,
    )
    ex_prior = m.prior.contrastive_div ? mean(logprior) : zero(T)
    logllhood, st_gen =
        log_likelihood_IS(z, x, m.lkhood, ps.gen, st.gen; rng = rng, ε = m.ε)

    loss, B = zero(T), size(x)[end]
    for b = 1:B
        loss +=
            dot(weights_resampled[b, :], logprior[resampled_idxs[b, :]]) +
            dot(weights_resampled[b, :], logllhood[b, resampled_idxs[b, :]])
    end

    return -((loss / B) - ex_prior)*m.loss_scaling, st_ebm, st_gen
end

function importance_loss(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    st::NamedTuple,
    model::Any,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    """MLE loss with importance sampling."""

    z, st_ebm, st_gen, weights_resampled, resampled_idxs =
        sample_importance(ps, st, model, x; rng = rng)

    f =
        (p, z_i, x_i, w, r, m, se, sg) -> begin
            first(marginal_llhood(p, z_i, x_i, w, r, m, se, sg; rng = rng))
        end

    CUDA.@fastmath Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        f,
        Enzyme.Active,
        Enzyme.Duplicated(ps, ∇),
        Enzyme.Const(z),
        Enzyme.Const(x),
        Enzyme.Const(weights_resampled),
        Enzyme.Const(resampled_idxs),
        Enzyme.Const(model),
        Enzyme.Const(st_ebm),
        Enzyme.Const(st_gen),
    )

    loss, st_ebm, st_gen = marginal_llhood(
        ps,
        z,
        x,
        weights_resampled,
        resampled_idxs,
        model,
        st_ebm,
        st_gen;
        rng = rng,
    )
    return loss, ∇, st_ebm, st_gen
end

end
