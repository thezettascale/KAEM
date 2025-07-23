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
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    m,
    x::AbstractArray{T},
    train_idx::Int,
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple} where {T<:half_quant}
    temps = collect(T, [(k / model.N_t)^model.p[train_idx] for k = 0:model.N_t])
    z, st_lux = m.posterior_sample(m, x, temps[2:end], ps, st_kan, st_lux, rng)
    Δt = device(temps[2:end] - temps[1:(end-1)])
    return z, Δt, st_lux
end

function marginal_llhood(
    ps::ComponentArray{T},
    z_posterior::AbstractArray{T},
    z_prior::AbstractArray{T},
    x::AbstractArray{T},
    Δt::AbstractVector{T},
    m,
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple;
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    Q, P, S, num_temps = size(z_posterior)
    log_ss = zero(T)

    # Steppingstone estimator
    x_rep = ndims(x) == 4 ? repeat(x, 1, 1, 1, num_temps) : repeat(x, 1, 1, num_temps)
    ll, st_gen = log_likelihood_MALA(
        reshape(z_posterior, Q, P, S*num_temps),
        x_rep,
        m.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux_gen;
        ε = m.ε,
    )
    log_ss = sum(mean(reshape(ll, num_temps, S) .* Δt; dims = 2))

    # MLE estimator
    logprior_pos, st_ebm = m.prior.lp_fcn(
        z_posterior[:, :, :, num_temps-1],
        m.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm;
        ε = m.ε,
        normalize = !m.prior.contrastive_div,
    )

    logprior, st_ebm = m.prior.lp_fcn(
        z_prior,
        m.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux_ebm;
        ε = m.ε,
        normalize = !m.prior.contrastive_div,
    )
    ex_prior = m.prior.contrastive_div ? mean(logprior) : zero(T)

    logllhood, st_gen =
        log_likelihood_MALA(z_prior[:, :, :, 1], x, m.lkhood, ps.gen, st_kan.gen, st_lux_gen; ε = m.ε)
    steppingstone_loss = mean(logllhood .* view(Δt, 1)) + log_ss
    return -(steppingstone_loss + mean(logprior_pos) - ex_prior) * m.loss_scaling,
    st_ebm,
    st_gen
end

function grad_thermo_llhood(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    z_posterior::AbstractArray{T},
    z_prior::AbstractArray{T},
    x::AbstractArray{T},
    Δt::AbstractVector{T},
    model,
    st_kan::ComponentArray{T},
    st_lux_ebm::NamedTuple,
    st_lux_gen::NamedTuple;
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    f =
        (p, post_i, prior_i, x_i, t, m, se, sg) -> begin
            first(marginal_llhood(p, post_i, prior_i, x_i, t, m, se, sg))
        end

    CUDA.@fastmath Enzyme.autodiff(
        Enzyme.Reverse,
        f,
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

struct ThermodynamicLoss
    compiled_loss::Any
    compiled_grad::Any
end

function initialize_thermo_loss(
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    model,
    x::AbstractArray{T};
    compile_mlir::Bool = false,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:half_quant}
    ∇ = Enzyme.make_zero(ps)
    z_posterior, Δt, st_lux = sample_thermo(ps, st_kan, st_lux, model, x, 1; rng = rng)
    st_ebm, st_gen = st_lux.ebm, st_lux.gen
    z_prior, st_ebm = model.prior.sample_z(model, size(x)[end], ps, st_kan, st_lux, rng)
    compiled_loss = marginal_llhood
    compiled_grad = grad_thermo_llhood

    if compile_mlir
        compiled_loss = Reactant.@compile marginal_llhood(
            ps,
            z_posterior,
            z_prior,
            x,
            Δt,
            model,
            Lux.testmode(st_ebm),
            Lux.testmode(st_gen),
        )
        compiled_grad = Reactant.@compile grad_thermo_llhood(
            ps,
            ∇,
            z_posterior,
            z_prior,
            x,
            Δt,
            model,
            Lux.trainmode(st_ebm),
            Lux.trainmode(st_gen),
        )
    end

    return ThermodynamicLoss(compiled_loss, compiled_grad)
end

function thermodynamic_loss(
    l,
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    model,
    x::AbstractArray{T};
    train_idx::Int=1,
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{T,AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    z_posterior, Δt, st_lux =
        sample_thermo(ps, st_kan, Lux.testmode(st_lux), model, x, train_idx; rng = rng)
    st_ebm, st_gen = st_lux.ebm, st_lux.gen
    z_prior, st_ebm = model.prior.sample_z(model, size(x)[end], ps, st_kan, st_lux, rng)

    ∇, st_ebm, st_gen = l.compiled_grad(
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
    loss, st_ebm, st_gen = l.compiled_loss(
        ps,
        z_posterior,
        z_prior,
        x,
        Δt,
        model,
        st_kan,
        Lux.testmode(st_lux_ebm),
        Lux.testmode(st_lux_gen),
    )
    return loss, ∇, st_ebm, st_gen
end

end
