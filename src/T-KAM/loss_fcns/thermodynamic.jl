module ThermodynamicIntegration

export initialize_thermo_loss, ThermodynamicLoss

using CUDA, ComponentArrays, Random, Zygote
using Statistics, Lux, LuxCUDA

using ..Utils
using ..T_KAM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

function sample_thermo(
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    model::T_KAM{T,full_quant},
    x::AbstractArray{T};
    train_idx::Int = 1,
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{
    AbstractArray{T,4},
    AbstractArray{T,1},
    NamedTuple,
    AbstractArray{T},
    AbstractArray{T},
} where {T<:half_quant}
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

    Δt = pu(temps[2:end] - temps[1:(end-1)])
    tempered_noise = randn(rng, T, model.lkhood.x_shape..., prod(size(z)[3:4])) |> pu
    noise = randn(rng, T, model.lkhood.x_shape..., size(x)[end]) |> pu
    return z, Δt, st_lux, noise, tempered_noise
end

function marginal_llhood(
    ps::ComponentArray{T},
    z_posterior::AbstractArray{T,4},
    z_prior::AbstractArray{T,3},
    x::AbstractArray{T},
    Δt::AbstractArray{T,1},
    model::T_KAM{T,full_quant},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple,
    noise::AbstractArray{T},
    tempered_noise::AbstractArray{T};
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    Q, P, S, num_temps = size(z_posterior)
    log_ss = zero(T)

    # Steppingstone estimator
    x_rep = model.lkhood.SEQ ? repeat(x, 1, 1, num_temps) : repeat(x, 1, 1, 1, num_temps)
    ll, st_lux_gen = log_likelihood_MALA(
        reshape(z_posterior, Q, P, S*num_temps),
        x_rep,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux_gen,
        tempered_noise;
        ε = model.ε,
    )
    log_ss = sum(reshape(ll, num_temps, S) .* Δt) / S

    # MLE estimator
    logprior_pos, st_lux_ebm = model.log_prior(
        z_posterior[:, :, :, num_temps-1],
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm,
    )

    logprior, st_lux_ebm =
        model.log_prior(z_prior, model.prior, ps.ebm, st_kan.ebm, st_lux_ebm)
    ex_prior = model.prior.bool_config.contrastive_div ? mean(logprior) : zero(T)

    logllhood, st_lux_gen = log_likelihood_MALA(
        z_prior,
        x,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux_gen,
        noise;
        ε = model.ε,
    )
    steppingstone_loss = mean(logllhood .* view(Δt, 1)) + log_ss
    return -(steppingstone_loss + mean(logprior_pos) - ex_prior) *
           model.loss_scaling.reduced,
    st_lux_ebm,
    st_lux_gen
end

function closure(
    ps::ComponentArray{T},
    z_posterior::AbstractArray{T,4},
    z_prior::AbstractArray{T,3},
    x::AbstractArray{T},
    Δt::AbstractArray{T,1},
    model::T_KAM{T,full_quant},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple,
    noise::AbstractArray{T},
    tempered_noise::AbstractArray{T};
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
            noise,
            tempered_noise,
        ),
    )
end

function grad_thermo_llhood(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    z_posterior::AbstractArray{T,4},
    z_prior::AbstractArray{T,3},
    x::AbstractArray{T},
    Δt::AbstractArray{T,1},
    model::T_KAM{T,full_quant},
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple,
    noise::AbstractArray{T},
    tempered_noise::AbstractArray{T};
)::AbstractArray{T} where {T<:half_quant}

    f =
        p -> closure(
            p,
            z_posterior,
            z_prior,
            x,
            Δt,
            model,
            st_kan,
            st_lux_ebm,
            st_lux_gen,
            noise,
            tempered_noise,
        )

    return CUDA.@fastmath first(Zygote.gradient(f, ps))
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
    z_posterior, Δt, st_lux, noise, tempered_noise = sample_thermo(
        ps,
        st_kan,
        Lux.testmode(st_lux),
        model,
        x;
        train_idx = train_idx,
        rng = rng,
    )
    st_lux_ebm, st_lux_gen = st_lux.ebm, st_lux.gen
    z_prior, st_ebm =
        model.sample_prior(model, size(x)[end], ps, st_kan, Lux.testmode(st_lux), rng)

    ∇ .= grad_thermo_llhood(
        ps,
        z_posterior,
        z_prior,
        x,
        Δt,
        model,
        st_kan,
        Lux.trainmode(st_lux_ebm),
        Lux.trainmode(st_lux_gen),
        noise,
        tempered_noise,
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
        noise,
        tempered_noise,
    )
    return loss, ∇, st_lux_ebm, st_lux_gen
end

end
