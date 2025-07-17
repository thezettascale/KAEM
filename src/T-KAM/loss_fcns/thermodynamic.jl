module ThermodynamicIntegration

export thermo_loss

using CUDA, KernelAbstractions, Enzyme, ComponentArrays, Random
using Statistics, Lux, LuxCUDA

include("../../utils.jl")
using .Utils: device, half_quant, full_quant, hq

function sample_thermo(
    ps::ComponentArray{T},
    st::NamedTuple,
    m::Any,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple} where {T<:half_quant}
    temps = collect(T, [(k / m.N_t)^m.p[st.train_idx] for k = 0:m.N_t])
    z, st = m.posterior_sample(m, x, temps[2:end], ps, st, rng)
    return z, temps, st
end

function marginal_llhood(
    ps::ComponentArray{T},
    z_posterior::AbstractArray{T},
    z_prior::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractVector{T},
    m::Any,
    st_ebm::NamedTuple,
    st_gen::NamedTuple;
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    log_ss = zero(T)
    st_ebm, st_gen = st_new.ebm, st_new.gen

    # Steppingstone estimator
    for k = 1:(T_length-2)
        logllhood, st_gen = log_likelihood_MALA(
            z_posterior[:, :, :, k],
            x,
            m.lkhood,
            ps.gen,
            st_gen;
            ε = m.ε,
        )
        log_ss += mean(logllhood .* Δt[k+1])
    end

    # MLE estimator
    logprior, st_ebm = m.prior.lp_fcn(
        z_posterior[:, :, :, T_length-1],
        m.prior,
        ps.ebm,
        st_ebm;
        ε = m.ε,
        normalize = !m.prior.contrastive_div,
    )
    contrastive_div = mean(logprior)

    if m.prior.contrastive_div
        logprior, st_ebm = m.prior.lp_fcn(
            z_prior,
            m.prior,
            ps.ebm,
            st_ebm;
            ε = m.ε,
            normalize = !m.prior.contrastive_div,
        )
        contrastive_div -= mean(logprior)
    end

    logllhood, st_gen =
        log_likelihood_MALA(m.lkhood, ps.gen, st_gen, x, z_prior[:, :, :, 1]; ε = m.ε)
    log_ss += mean(logllhood .* Δt[1])

    return -(log_ss + contrastive_div) * m.loss_scaling, st_ebm, st_gen
end

function grad_thermo_llhood(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    z_posterior::AbstractArray{T},
    z_prior::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractVector{T},
    m::Any,
    st_ebm::NamedTuple,
    st_gen::NamedTuple;
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    f =
        (p, post_i, prior_i, x_i, t, m, se, sg) -> begin
            first(marginal_llhood(p, post_i, prior_i, x_i, t, m, se, sg))
        end

    CUDA.@fastmath Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        f,
        Enzyme.Active,
        Enzyme.Duplicated(ps, ∇),
        Enzyme.Const(z_posterior),
        Enzyme.Const(z_prior),
        Enzyme.Const(x),
        Enzyme.Const(temps),
        Enzyme.Const(model),
        Enzyme.Const(st_ebm),
        Enzyme.Const(st_gen),
    )
    return ∇, st_ebm, st_gen
end

function thermo_loss(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    st::NamedTuple,
    model::Any,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    z_posterior, temps, st = sample_thermo(ps, st, model, x; rng = rng)
    st_ebm, st_gen = st.ebm, st.gen
    z_prior, st_ebm = m.prior.sample_z(m, size(x)[end], ps, st, rng)
    Δt, T_length, B = temps[2:end] - temps[1:(end-1)], length(temps), size(x)[end]

    ∇, st_ebm, st_gen =
        grad_thermo_llhood(ps, ∇, z_posterior, z_prior, x, temps, model, st_ebm, st_gen)

    loss, st_ebm, st_gen =
        marginal_llhood(ps, z_posterior, z_prior, x, temps, model, st_ebm, st_gen)
    return loss, ∇, st_ebm, st_gen
end

end
