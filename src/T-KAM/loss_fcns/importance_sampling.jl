module ImportanceSampling

using CUDA, KernelAbstractions, Enzyme, ComponentArrays
using Statistics, Lux, LuxCUDA
using NNlib: softmax

export importance_loss

include("../../utils.jl")
using .Utils: device, next_rng, half_quant, full_quant, hq

function sample_importance(
    ps::ComponentArray{T},
    st::NamedTuple,
    m::Any,
    x::AbstractArray{T};
    seed::Int = 1,
)::Tuple{
    AbstractArray{T},
    NamedTuple,
    NamedTuple,
    AbstractArray{Int},
    Int,
} where {T<:half_quant}
    z, st_ebm, seed = m.prior.sample_z(m, m.IS_samples, ps, st, seed)
    logllhood, st_gen, seed =
        log_likelihood_IS(z, x, m.lkhood, ps.gen, st.gen; seed = seed, ε = m.ε)
    weights = softmax(full_quant.(logllhood), dims = 2)
    resampled_idxs, seed = m.lkhood.resample_z(weights, seed)
    weights_resampled =
        softmax(
            reduce(vcat, map(b -> weights[b:b, resampled_idxs[b, :]], 1:size(x)[end])),
            dims = 2,
        ) .|> T
    return z, st_ebm, st_gen, weights_resampled, resampled_idxs, seed
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
    seed::Int = 1,
)::Tuple{T,NamedTuple,NamedTuple,Int} where {T<:half_quant}
    logprior, st_ebm = m.prior.lp_fcn(
        z,
        m.prior,
        ps.ebm,
        st_ebm;
        ε = m.ε,
        normalize = !m.prior.contrastive_div,
    )
    ex_prior = m.prior.contrastive_div ? mean(logprior) : zero(T)
    logllhood, st_gen, seed =
        log_likelihood_IS(z, x, m.lkhood, ps.gen, st.gen; seed = seed, ε = m.ε)

    # Parse samples
    logprior_resampled =
        reduce(hcat, map(b -> logprior[resampled_idxs[b, :], :], 1:size(x)[end]))
    logllhood_resampled =
        reduce(vcat, map(b -> logllhood[b:b, resampled_idxs[b, :]], 1:size(x)[end]))

    # Expected posterior
    loss_prior = sum(weights_resampled .* logprior_resampled', dims = 2)
    loss_llhood = sum(weights_resampled .* logllhood_resampled, dims = 2)

    return -(mean(loss_prior .+ loss_llhood) - ex_prior)*m.loss_scaling,
    st_ebm,
    st_gen,
    seed
end

function importance_loss(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    st::NamedTuple,
    model::Any,
    x::AbstractArray{T};
    seed::Int = 1,
)::Tuple{T,NamedTuple,NamedTuple,Int} where {T<:half_quant}
    """MLE loss with importance sampling."""

    z, st_ebm, st_gen, weights_resampled, resampled_idxs, seed =
        sample_importance(ps, st, model, x; seed = seed)

    f =
        (p, z_i, x_i, w, r, m, se, sg) -> begin
            first(marginal_llhood(p, z_i, x_i, w, r, m, se, sg; seed = seed))
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

    loss, st_ebm, st_gen, seed = marginal_llhood(
        ps,
        z,
        x,
        weights_resampled,
        resampled_idxs,
        model,
        st_ebm,
        st_gen;
        seed = seed,
    )
    return loss, ∇, st_ebm, st_gen, seed
end

end
