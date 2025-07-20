module ThermodynamicIntegration

export initialize_thermo_loss, thermodynamic_loss

using CUDA, KernelAbstractions, Enzyme, ComponentArrays, Random, Reactant
using Statistics, Lux, LuxCUDA

include("../gen/loglikelihoods.jl")
include("../../utils.jl")
using .LogLikelihoods: log_likelihood_MALA
using .Utils: device, half_quant, full_quant, hq

function sample_thermo(
    ps::ComponentArray{T},
    st::NamedTuple,
    m,
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
    m,
    st_ebm::NamedTuple,
    st_gen::NamedTuple;
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    num_temps = length(temps)
    Δt = temps[2:end] - temps[1:(end-1)]
    log_ss = zeros(T, num_temps-1)

    # Steppingstone estimator
    for k = 1:(num_temps-2)
        logllhood, st_gen = log_likelihood_MALA(
            z_posterior[:, :, :, k],
            x,
            m.lkhood,
            ps.gen,
            st_gen;
            ε = m.ε,
        )
        log_ss[k] = mean(logllhood .* Δt[k+1])
    end

    # MLE estimator
    logprior_pos, st_ebm = m.prior.lp_fcn(
        z_posterior[:, :, :, num_temps-1],
        m.prior,
        ps.ebm,
        st_ebm;
        ε = m.ε,
        normalize = !m.prior.contrastive_div,
    )

    logprior, st_ebm = m.prior.lp_fcn(
        z_prior,
        m.prior,
        ps.ebm,
        st_ebm;
        ε = m.ε,
        normalize = !m.prior.contrastive_div,
    )
    ex_prior = m.prior.contrastive_div ? mean(logprior) : zero(T)

    logllhood, st_gen =
        log_likelihood_MALA(z_prior[:, :, :, 1], x, m.lkhood, ps.gen, st_gen; ε = m.ε)
    steppingstone_loss = mean(logllhood .* Δt[1]) + sum(log_ss)
    return -(steppingstone_loss + mean(logprior_pos) - ex_prior) * m.loss_scaling, st_ebm, st_gen
end

function grad_thermo_llhood(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    z_posterior::AbstractArray{T},
    z_prior::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractVector{T},
    model,
    st_ebm::NamedTuple,
    st_gen::NamedTuple;
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
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

struct ThermodynamicLoss
    compiled_loss::Any
    compiled_grad::Any
end

function initialize_thermo_loss(
    ps::ComponentArray{T},
    st::NamedTuple,
    model,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
) where {T<:half_quant}
    ∇ = Enzyme.make_zero(ps)
    z_posterior, temps, st = sample_thermo(ps, Lux.testmode(st), model, x; rng = rng)
    st_ebm, st_gen = st.ebm, st.gen
    z_prior, st_ebm = model.prior.sample_z(model, size(x)[end], ps, Lux.testmode(st), rng)

    compiled_loss = Reactant.@compile marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        temps,
        model,
        st_ebm,
        st_gen,
    )
    compiled_grad = Reactant.@compile grad_thermo_llhood(
        ps,
        ∇,
        z_posterior,
        z_prior,
        x,
        temps,
        model,
        Lux.trainmode(st_ebm),
        Lux.trainmode(st_gen),
    )
    return ThermodynamicLoss(compiled_loss, compiled_grad)
end

function thermodynamic_loss(
    l,
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    st::NamedTuple,
    model,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{T,AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    z_posterior, temps, st = sample_thermo(ps, Lux.testmode(st), model, x; rng = rng)
    st_ebm, st_gen = st.ebm, st.gen
    z_prior, st_ebm = model.prior.sample_z(model, size(x)[end], ps, Lux.testmode(st), rng)

    ∇, st_ebm, st_gen = l.compiled_grad(
        ps,
        ∇,
        z_posterior,
        z_prior,
        x,
        temps,
        model,
        Lux.trainmode(st_ebm),
        Lux.trainmode(st_gen),
    )
    loss, st_ebm, st_gen =
        l.compiled_loss(ps, z_posterior, z_prior, x, temps, model, st_ebm, st_gen)
    return loss, ∇, st_ebm, st_gen
end

end
