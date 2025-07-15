module MarginalLikelihood

export importance_loss, langevin_loss, thermo_loss

using CUDA, KernelAbstractions, Enzyme.EnzymeRules
using Statistics, Lux, LuxCUDA
using NNlib: softmax

include("../utils.jl")
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

function sample_langevin(
    ps::ComponentArray{T},
    st::NamedTuple,
    m::Any,
    x::AbstractArray{T};
    seed::Int = 1,
)::Tuple{AbstractArray{T},NamedTuple,Int} where {T<:half_quant}
    z, st_ebm, seed = m.posterior_sample(m, x, 0, ps, st, seed)
    return z, st_ebm, seed
end

function sample_thermo(
    ps::ComponentArray{T},
    st::NamedTuple,
    m::Any,
    x::AbstractArray{T};
    seed::Int = 1,
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple,Int} where {T<:half_quant}
    temps = collect(T, [(k / m.N_t)^m.p[st.train_idx] for k = 0:m.N_t])
    z, st, seed = m.posterior_sample(m, x, temps[2:end], ps, st, seed)
    return z, temps, st, seed
end

EnzymeRules.inactive(::typeof(sample_importance), args...) = nothing
EnzymeRules.inactive(::typeof(sample_langevin), args...) = nothing
EnzymeRules.inactive(::typeof(sample_thermo), args...) = nothing

function importance_loss(
    ps::ComponentArray{T},
    st::NamedTuple,
    m::Any,
    x::AbstractArray{T};
    seed::Int = 1,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple,Int} where {T<:half_quant}
    """MLE loss with importance sampling."""

    z, st_ebm, st_gen, weights_resampled, resampled_idxs, seed =
        sample_importance(ps, st, m, x; seed = seed)

    # Log-dists
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

function langevin_loss(
    ps::ComponentArray{T},
    st::NamedTuple,
    m::Any,
    x::AbstractArray{T};
    seed::Int = 1,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple,Int} where {T<:half_quant}
    """MLE loss without importance, (used when posterior expectation = MCMC estimate)."""

    z, st_new, seed = sample_langevin(ps, st, m, x; seed = seed)
    st_ebm, st_gen = st_new.ebm, st_new.gen

    # Log-dists
    logprior_pos, st_ebm = m.prior.lp_fcn(
        z[:, :, :, 1],
        m.prior,
        ps.ebm,
        st_ebm;
        ε = m.ε,
        normalize = !m.prior.contrastive_div,
    )
    logllhood, st_gen, seed = log_likelihood_MALA(
        z[:, :, :, 1],
        x,
        m.lkhood,
        ps.gen,
        st_gen;
        seed = seed,
        ε = m.ε,
    )
    contrastive_div = mean(logprior_pos)

    if m.prior.contrastive_div
        z, st_ebm, seed = m.prior.sample_z(m, size(x)[end], ps, st, seed)
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

    return -(contrastive_div + mean(logllhood))*m.loss_scaling, st_ebm, st_gen, seed
end
function thermo_loss(
    ps::ComponentArray{T},
    st::NamedTuple,
    m::Any,
    x::AbstractArray{T};
    seed::Int = 1,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple,Int} where {T<:half_quant}
    """Thermodynamic integration loss."""

    # Schedule temperatures
    z, temps, st_new, seed = sample_thermo(ps, st, m, x; seed = seed)
    Δt, T_length, B = temps[2:end] - temps[1:(end-1)], length(temps), size(x)[end]

    log_ss = zero(T)
    st_ebm, st_gen = st_new.ebm, st_new.gen

    # Steppingstone estimator
    for k = 1:(T_length-2)
        logllhood, st_gen, seed = log_likelihood_MALA(
            z[:, :, :, k],
            x,
            m.lkhood,
            ps.gen,
            st_gen;
            seed = seed,
            ε = m.ε,
        )
        log_ss += mean(logllhood .* Δt[k+1])
    end

    # MLE estimator
    logprior, st_ebm = m.prior.lp_fcn(
        z[:, :, :, T_length-1],
        m.prior,
        ps.ebm,
        st_ebm;
        ε = m.ε,
        normalize = !m.prior.contrastive_div,
    )
    contrastive_div = mean(logprior)

    z, st_ebm, seed = m.prior.sample_z(m, B, ps, st, seed)
    if m.prior.contrastive_div
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

    logllhood, st_gen, seed = log_likelihood_MALA(
        m.lkhood,
        ps.gen,
        st_gen,
        x,
        z[:, :, :, 1];
        seed = seed,
        ε = m.ε,
    )
    log_ss += mean(logllhood .* Δt[1])

    return -(log_ss + contrastive_div) * m.loss_scaling, st_ebm, st_gen, seed
end

end
