module ThermodynamicIntegration

export initialize_thermo_loss, ThermodynamicLoss

using CUDA, KernelAbstractions, Enzyme, ComponentArrays, Random
using Statistics, Lux, LuxCUDA

include("../gen/loglikelihoods.jl")
include("../T-KAM.jl")
include("../../utils.jl")
using .T_KAM_model
using .LogLikelihoods: log_likelihood_MALA
using .Utils: device, half_quant, full_quant, hq

function sample_thermo(
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    model::T_KAM{T,full_quant},
    x::AbstractArray{T};
    train_idx::Int = 1,
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple} where {T<:half_quant}
    temps = collect(T, [(k / model.N_t)^model.p[train_idx] for k = 0:model.N_t])
    z, st_lux = model.posterior_sampler(
        model,
        ps,
        st_kan,
        st_lux,
        x;
        temps = temps[2:end],
        rng = rng,
    )
    Δt = device(temps[2:end] - temps[1:(end-1)])
    return z, Δt, st_lux
end

function marginal_llhood(
    ps::ComponentArray{T},
    z_posterior::AbstractArray{T},
    z_prior::AbstractArray{T},
    x::AbstractArray{T},
    Δt::AbstractVector{T},
    model::T_KAM{T,full_quant},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple;
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    Q, P, S, num_temps = size(z_posterior)
    log_ss = zero(T)

    # Steppingstone estimator
    x_rep = ndims(x) == 4 ? repeat(x, 1, 1, 1, num_temps) : repeat(x, 1, 1, num_temps)
    ll, st_lux_gen = log_likelihood_MALA(
        reshape(z_posterior, Q, P, S*num_temps),
        x_rep,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux_gen;
        ε = model.ε,
    )
    log_ss = sum(mean(reshape(ll, num_temps, S) .* Δt; dims = 2))

    # MLE estimator
    logprior_pos, st_lux_ebm = model.prior.lp_fcn(
        z_posterior[:, :, :, num_temps-1],
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm;
        ε = model.ε,
        normalize = !model.prior.contrastive_div,
    )

    logprior, st_lux_ebm = model.prior.lp_fcn(
        z_prior,
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm;
        ε = model.ε,
        normalize = !model.prior.contrastive_div,
    )
    ex_prior = model.prior.contrastive_div ? mean(logprior) : zero(T)

    logllhood, st_lux_gen = log_likelihood_MALA(
        z_prior[:, :, :, 1],
        x,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux_gen;
        ε = model.ε,
    )
    steppingstone_loss = mean(logllhood .* view(Δt, 1)) + log_ss
    return -(steppingstone_loss + mean(logprior_pos) - ex_prior) * model.loss_scaling,
    st_lux_ebm,
    st_lux_gen
end

function closure(
    ps::ComponentArray{T},
    z_posterior::AbstractArray{T},
    z_prior::AbstractArray{T},
    x::AbstractArray{T},
    Δt::AbstractVector{T},
    model::T_KAM{T,full_quant},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple;
)::T where {T<:half_quant}
    return first(
        marginal_llhood(
            ps,
            z_posterior,
            z_prior,
            x,
            Δt,
            model,
            st_kan,
            st_lux_ebm,
            st_lux_gen,
        ),
    )
end

function grad_thermo_llhood(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    z_posterior::AbstractArray{T},
    z_prior::AbstractArray{T},
    x::AbstractArray{T},
    Δt::AbstractVector{T},
    model::T_KAM{T,full_quant},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple;
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    CUDA.@fastmath Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        closure,
        Enzyme.Active,
        Enzyme.Duplicated(ps, ∇),
        Enzyme.Const(z_posterior),
        Enzyme.Const(z_prior),
        Enzyme.Const(x),
        Enzyme.Const(Δt),
        Enzyme.Const(model),
        Enzyme.Const(st_kan),
        Enzyme.Const(st_lux_ebm),
        Enzyme.Const(st_lux_gen),
    )

    return ∇, st_lux_ebm, st_lux_gen
end

struct ThermodynamicLoss end

function (l::ThermodynamicLoss)(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    model::T_KAM{T,full_quant},
    x::AbstractArray{T};
    train_idx::Int = 1,
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{T,AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    z_posterior, Δt, st_lux = sample_thermo(
        ps,
        st_kan,
        Lux.testmode(st_lux),
        model,
        x;
        train_idx = train_idx,
        rng = rng,
    )
    st_lux_ebm, st_lux_gen = st_lux.ebm, st_lux.gen
    z_prior, st_ebm = model.sample_prior(model, size(x)[end], ps, st_kan, st_lux, rng)

    ∇, st_lux_ebm, st_lux_gen = grad_thermo_llhood(
        ps,
        ∇,
        z_posterior,
        z_prior,
        x,
        Δt,
        model,
        st_kan,
        Lux.trainmode(st_lux_ebm),
        Lux.trainmode(st_lux_gen),
    )
    loss, st_lux_ebm, st_lux_gen = marginal_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        Δt,
        model,
        st_kan,
        Lux.trainmode(st_lux_ebm),
        Lux.trainmode(st_lux_gen),
    )
    return loss, ∇, st_lux_ebm, st_lux_gen
end

end
