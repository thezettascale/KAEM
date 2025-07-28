module ImportanceSampling

using CUDA, Enzyme, ComponentArrays, Random
using Statistics, Lux, LuxCUDA, ParallelStencil
using NNlib: softmax

export ImportanceLoss, initialize_importance_loss

using ..Utils
using ..T_KAM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_IS

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, half_quant, 3)
else
    @init_parallel_stencil(Threads, half_quant, 3)
end

function sample_importance(
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    m::T_KAM{T,full_quant},
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{
    AbstractArray{T},
    NamedTuple,
    NamedTuple,
    AbstractArray{T},
    AbstractArray{Int},
} where {T<:half_quant,U<:full_quant}
    z, st_lux_ebm = m.sample_prior(m, m.IS_samples, ps, st_kan, st_lux, rng)
    noise = pu(randn(rng, T, m.lkhood.x_shape..., size(z)[end], size(x)[end]))
    logllhood, st_lux_gen =
        log_likelihood_IS(z, x, m.lkhood, ps.gen, st_kan.gen, st_lux.gen, noise; ε = m.ε)
    weights = softmax(U.(logllhood), dims = 2)
    resampled_idxs = m.lkhood.resample_z(weights, rng)
    weights = T.(weights)
    weights_resampled = softmax(
        reduce(vcat, map(b -> weights[b:b, resampled_idxs[b, :]], 1:size(x)[end])),
        dims = 2,
    )
    return z, st_lux_ebm, st_lux_gen, weights_resampled, resampled_idxs
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
    m::T_KAM{T},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple,
    zero_vec::AbstractArray{T},
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    logprior, st_lux_ebm = m.log_prior(z, m.prior, ps.ebm, st_kan.ebm, st_lux_ebm)
    ex_prior = m.prior.contrastive_div ? mean(logprior) : zero(T)
    logllhood, st_gen =
        log_likelihood_IS(z, x, m.lkhood, ps.gen, st_kan.gen, st_lux_gen, zero_vec; ε = m.ε)

    B, S = size(x)[end], size(z)[end]
    marginal_llhood = @zeros(B, S)
    @parallel (1:B, 1:S) resampled_kernel!(
        marginal_llhood,
        weights_resampled,
        logprior,
        logllhood,
        resampled_idxs,
    )
    return -(mean(sum(marginal_llhood, dims = 2)) - ex_prior)*m.loss_scaling,
    st_lux_ebm,
    st_lux_gen
end

function closure(
    ps::ComponentArray{T},
    z::AbstractArray{T},
    x::AbstractArray{T},
    weights_resampled::AbstractArray{T},
    resampled_idxs::AbstractArray{Int},
    m::T_KAM{T},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple,
    zero_vec::AbstractArray{T},
)::T where {T<:half_quant}
    return first(
        marginal_llhood(
            ps,
            z,
            x,
            weights_resampled,
            resampled_idxs,
            m,
            st_kan,
            st_lux_ebm,
            st_lux_gen,
            zero_vec,
        ),
    )
end

function grad_importance_llhood(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    z::AbstractArray{T},
    x::AbstractArray{T},
    weights_resampled::AbstractArray{T},
    resampled_idxs::AbstractArray{Int},
    model::T_KAM{T,full_quant},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple,
    zero_vec::AbstractArray{T},
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    CUDA.@fastmath Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(closure),
        Enzyme.Active,
        Enzyme.Duplicated(ps, ∇),
        Enzyme.Const(z),
        Enzyme.Const(x),
        Enzyme.Const(weights_resampled),
        Enzyme.Const(resampled_idxs),
        Enzyme.Const(model),
        Enzyme.Const(st_kan),
        Enzyme.Const(st_lux_ebm),
        Enzyme.Const(st_lux_gen),
        Enzyme.Const(zero_vec),
    )

    return ∇
end

struct ImportanceLoss
    zero_vec::AbstractArray{half_quant}
end

function (l::ImportanceLoss)(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    model::T_KAM{T,full_quant},
    x::AbstractArray{T};
    train_idx::Int = 1,
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{T,AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    z, st_lux_ebm, st_lux_gen, weights_resampled, resampled_idxs =
        sample_importance(ps, st_kan, Lux.testmode(st_lux), model, x; rng = rng)

    ∇ = grad_importance_llhood(
        ps,
        ∇,
        z,
        x,
        weights_resampled,
        resampled_idxs,
        model,
        st_kan,
        Lux.trainmode(st_lux_ebm),
        Lux.trainmode(st_lux_gen),
        l.zero_vec,
    )

    loss, st_lux_ebm, st_lux_gen = marginal_llhood(
        ps,
        z,
        x,
        weights_resampled,
        resampled_idxs,
        model,
        st_kan,
        Lux.trainmode(st_lux_ebm),
        Lux.trainmode(st_lux_gen),
        l.zero_vec,
    )

    return loss, ∇, st_lux_ebm, st_lux_gen
end

end
