module ImportanceSampling

using CUDA, Enzyme, ComponentArrays, Random, Zygote
using Statistics, Lux, LuxCUDA
using NNlib: softmax

export ImportanceLoss, initialize_importance_loss

using ..Utils
using ..T_KAM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_IS

if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    include("is_kernel_gpu.jl")
    using .IS_Kernel
else
    include("is_kernel.jl")
    using .IS_Kernel
end

function sample_importance(
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    m::T_KAM{T,U},
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{
    AbstractArray{T,3},
    AbstractArray{T,3},
    NamedTuple,
    NamedTuple,
    AbstractArray{T,2},
    AbstractArray{Int,2},
    AbstractArray{T},
} where {T<:half_quant,U<:full_quant}

    # Prior is proposal for importance sampling
    z_posterior, st_lux_ebm = m.sample_prior(m, m.IS_samples, ps, st_kan, st_lux, rng)
    noise = pu(randn(rng, T, m.lkhood.x_shape..., size(z_posterior)[end], size(x)[end]))
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
    weights = softmax(U.(logllhood), dims = 2)
    resampled_idxs = m.lkhood.resample_z(weights, rng)
    weights = T.(weights)
    weights_resampled = softmax(
        reduce(vcat, map(b -> weights[b:b, resampled_idxs[b, :]], 1:size(x)[end])),
        dims = 2,
    )

    # Works better with more samples
    z_prior, st_lux_ebm = m.sample_prior(m, m.IS_samples, ps, st_kan, st_lux, rng)
    return z_posterior,
    z_prior,
    st_lux_ebm,
    st_lux_gen,
    weights_resampled,
    resampled_idxs,
    noise
end

function marginal_llhood(
    ps::ComponentArray{T},
    z_posterior::AbstractArray{T,3},
    z_prior::AbstractArray{T,3},
    x::AbstractArray{T},
    weights_resampled::AbstractArray{T,2},
    resampled_idxs::AbstractArray{Int,2},
    m::T_KAM{T},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple,
    noise::AbstractArray{T},
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    B, S = size(x)[end], size(z_posterior)[end]

    logprior_posterior, st_lux_ebm =
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

    marginal_llhood =
        loss_accum(weights_resampled, logprior_posterior, logllhood, resampled_idxs, B, S)

    logprior_prior, st_lux_ebm =
        m.log_prior(z_prior, m.prior, ps.ebm, st_kan.ebm, st_lux_ebm)
    ex_prior = m.prior.bool_config.contrastive_div ? mean(logprior_prior) : zero(T)

    return -(mean(marginal_llhood) - ex_prior)*m.loss_scaling.reduced, st_lux_ebm, st_lux_gen
end

function closure(
    ps::ComponentArray{T},
    z_posterior::AbstractArray{T,3},
    z_prior::AbstractArray{T,3},
    x::AbstractArray{T},
    weights_resampled::AbstractArray{T,2},
    resampled_idxs::AbstractArray{Int,2},
    m::T_KAM{T,full_quant},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple,
    noise::AbstractArray{T},
)::T where {T<:half_quant}
    return first(
        marginal_llhood(
            ps,
            z_posterior,
            z_prior,
            x,
            weights_resampled,
            resampled_idxs,
            m,
            st_kan,
            st_lux_ebm,
            st_lux_gen,
            noise,
        ),
    )
end

function grad_importance_llhood(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    z_posterior::AbstractArray{T,3},
    z_prior::AbstractArray{T,3},
    x::AbstractArray{T},
    weights_resampled::AbstractArray{T,2},
    resampled_idxs::AbstractArray{Int,2},
    model::T_KAM{T,full_quant},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple,
    noise::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}

    if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
        f =
            p -> closure(
                p,
                z_posterior,
                z_prior,
                x,
                weights_resampled,
                resampled_idxs,
                model,
                st_kan,
                st_lux_ebm,
                st_lux_gen,
                noise,
            )
        ∇ = CUDA.@fastmath first(Zygote.gradient(f, ps))
    else
        CUDA.@fastmath Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            Enzyme.Const(closure),
            Enzyme.Active,
            Enzyme.Duplicated(ps, ∇),
            Enzyme.Const(z_posterior),
            Enzyme.Const(z_prior),
            Enzyme.Const(x),
            Enzyme.Const(weights_resampled),
            Enzyme.Const(resampled_idxs),
            Enzyme.Const(model),
            Enzyme.Const(st_kan),
            Enzyme.Const(st_lux_ebm),
            Enzyme.Const(st_lux_gen),
            Enzyme.Const(noise),
        )
    end

    return ∇
end

struct ImportanceLoss end

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

    z_posterior, z_prior, st_lux_ebm, st_lux_gen, weights_resampled, resampled_idxs, noise =
        sample_importance(ps, st_kan, Lux.testmode(st_lux), model, x; rng = rng)

    ∇ = grad_importance_llhood(
        ps,
        ∇,
        z_posterior,
        z_prior,
        x,
        weights_resampled,
        resampled_idxs,
        model,
        st_kan,
        Lux.trainmode(st_lux_ebm),
        Lux.trainmode(st_lux_gen),
        noise,
    )

    loss, st_lux_ebm, st_lux_gen = marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        weights_resampled,
        resampled_idxs,
        model,
        st_kan,
        Lux.trainmode(st_lux_ebm),
        Lux.trainmode(st_lux_gen),
        noise,
    )

    return loss, ∇, st_lux_ebm, st_lux_gen
end

end
