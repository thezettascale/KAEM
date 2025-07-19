module ImportanceSampling

using CUDA, KernelAbstractions, Enzyme, ComponentArrays, Random, Reactant
using Statistics, Lux, LuxCUDA, ParallelStencil
using NNlib: softmax

export initialize_importance_loss, importance_loss

include("../gen/loglikelihoods.jl")
include("../../utils.jl")
using .LogLikelihoods: log_likelihood_IS
using .Utils: device, half_quant, full_quant, hq

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

function sample_importance(
    ps::ComponentArray{T},
    st::NamedTuple,
    m,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{
    AbstractArray{T},
    NamedTuple,
    NamedTuple,
    AbstractArray{T},
    AbstractArray{Int},
} where {T<:half_quant}
    z, st_ebm = m.prior.sample_z(m, m.IS_samples, ps, st, rng)
    noise = device(randn(rng, T, m.lkhood.x_shape..., size(z)[end], size(x)[end]))
    logllhood, st_gen =
        log_likelihood_IS(z, x, m.lkhood, ps.gen, st.gen; ε = m.ε, noise = noise)
    weights = softmax(full_quant.(logllhood), dims = 2)
    resampled_idxs = m.lkhood.resample_z(weights, rng)
    weights = T.(weights)
    weights_resampled = softmax(
        reduce(vcat, map(b -> weights[b:b, resampled_idxs[b, :]], 1:size(x)[end])),
        dims = 2,
    )
    return z, st_ebm, st_gen, weights_resampled, resampled_idxs
end

@parallel_indices (b, s) function resampled_kernel!(
    loss::AbstractArray{T},
    weights_resampled::AbstractArray{T},
    logprior::AbstractArray{T},
    logllhood::AbstractArray{T},
    resampled_idxs::AbstractArray{Int},
)::Nothing where {T<:half_quant}
    idx = resampled_idxs[b, s]
    loss[b, s] = weights_resampled[b, s] * (logprior[idx] + logllhood[b, idx])
    return nothing
end

function marginal_llhood(
    ps::ComponentArray{T},
    z::AbstractArray{T},
    x::AbstractArray{T},
    weights_resampled::AbstractArray{T},
    resampled_idxs::AbstractArray{Int},
    m,
    st_ebm::NamedTuple,
    st_gen::NamedTuple;
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
    logllhood, st_gen = log_likelihood_IS(z, x, m.lkhood, ps.gen, st_gen; ε = m.ε)

    B, S = size(x)[end], size(z)[end]
    marginal_llhood = @zeros(B, S)
    @parallel (1:B, 1:S) resampled_kernel!(
        marginal_llhood,
        weights_resampled,
        logprior,
        logllhood,
        resampled_idxs,
    )
    return -(mean(sum(marginal_llhood, dims = 2)) - ex_prior)*m.loss_scaling, st_ebm, st_gen
end

function grad_importance_llhood(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    z::AbstractArray{T},
    x::AbstractArray{T},
    weights_resampled::AbstractArray{T},
    resampled_idxs::AbstractArray{Int},
    model,
    st_ebm::NamedTuple,
    st_gen::NamedTuple;
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    f =
        (p, z_i, x_i, w, r, m, se, sg) -> begin
            first(marginal_llhood(p, z_i, x_i, w, r, m, se, sg))
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

    return ∇, st_ebm, st_gen
end

struct ImportanceLoss
    compiled_loss
    compiled_grad
end

function initialize_importance_loss(
    ps::ComponentArray{T},
    st::NamedTuple,
    model,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
) where {T<:half_quant}
    ∇ = Enzyme.make_zero(ps)
    z, st_ebm, st_gen, weights_resampled, resampled_idxs =
        sample_importance(ps, st, model, x; rng = rng)
    compiled_loss = Reactant.@compile marginal_llhood(
        ps,
        z,
        x,
        weights_resampled,
        resampled_idxs,
        model,
        st_ebm,
        st_gen,
    )
    compiled_grad = Reactant.@compile grad_importance_llhood(
        ps,
        ∇,
        z,
        x,
        weights_resampled,
        resampled_idxs,
        model,
        st_ebm,
        st_gen,
    )
    return ImportanceLoss(marginal_llhood, grad_importance_llhood)
end

function importance_loss(
    l,
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    st::NamedTuple,
    model,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{T,AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    z, st_ebm, st_gen, weights_resampled, resampled_idxs =
        sample_importance(ps, st, model, x; rng = rng)
    ∇, st_ebm, st_gen = l.compiled_grad(
        ps,
        ∇,
        z,
        x,
        weights_resampled,
        resampled_idxs,
        model,
        st_ebm,
        st_gen,
    )
    loss, st_ebm, st_gen =
        l.compiled_loss(ps, z, x, weights_resampled, resampled_idxs, model, st_ebm, st_gen)
    return loss, ∇, st_ebm, st_gen
end

end
