module LogPosteriors

using CUDA, ComponentArrays, Statistics, Lux, LuxCUDA, LinearAlgebra, Random, Enzyme

using ..Utils
using ..T_KAM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

### ULA ### 
function unadjusted_logpos(
    z::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    prior_sampling_bool::Bool,
)::T where {T<:half_quant,U<:full_quant}
    Q, P, S, num_temps = size(z)
    z_reshaped = reshape(z, Q, P, S*num_temps)
    lp =
        sum(first(model.log_prior(z_reshaped, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)))
    ll = first(
        log_likelihood_MALA(
            z_reshaped,
            x,
            model.lkhood,
            ps.gen,
            st_kan.gen,
            st_lux.gen;
            ε = model.ε,
        ),
    )
    tempered_ll = sum(temps .* reshape(ll, num_temps, S))
    return (lp + tempered_ll) * model.loss_scaling
end

function unadjusted_logpos_grad(
    z::AbstractArray{T},
    ∇z::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    prior_sampling_bool::Bool,
)::AbstractArray{T} where {T<:half_quant,U<:full_quant}

    # Expand for log_likelihood
    x_expanded =
        ndims(x) == 4 ? repeat(x, 1, 1, 1, length(temps)) : repeat(x, 1, 1, length(temps))

    CUDA.@fastmath Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(unadjusted_logpos),
        Enzyme.Active,
        Enzyme.Duplicated(z, ∇z),
        Enzyme.Const(x_expanded),
        Enzyme.Const(temps),
        Enzyme.Const(model),
        Enzyme.Const(ps),
        Enzyme.Const(st_kan),
        Enzyme.Const(st_lux),
        Enzyme.Const(prior_sampling_bool),
    )

    return ∇z
end

### autoMALA ###
function autoMALA_logpos_value_4D(
    z::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant,U<:full_quant}
    Q, P, S, num_temps = size(z)
    z_reshaped = reshape(z, Q, P, S*num_temps)
    lp, st_ebm = model.log_prior(z_reshaped, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)
    ll, st_gen = log_likelihood_MALA(
        z_reshaped,
        x,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux.gen;
        ε = model.ε,
    )
    logpos = reshape(lp, S, num_temps) + temps .* reshape(ll, S, num_temps)
    return logpos .* model.loss_scaling, st_ebm, st_gen
end

function autoMALA_logpos_reduced_4D(
    z::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::T where {T<:half_quant,U<:full_quant}
    Q, P, S, num_temps = size(z)
    z_reshaped = reshape(z, Q, P, S*num_temps)
    lp =
        sum(first(model.log_prior(z_reshaped, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)))
    ll = first(
        log_likelihood_MALA(
            z_reshaped,
            x,
            model.lkhood,
            ps.gen,
            st_kan.gen,
            st_lux.gen;
            ε = model.ε,
        ),
    )
    tempered_ll = sum(temps .* reshape(ll, S, num_temps))
    return (lp + tempered_ll) * model.loss_scaling
end

function autoMALA_value_and_grad_4D(
    z::AbstractArray{T},
    ∇z::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::Tuple{
    AbstractArray{T},
    AbstractArray{T},
    NamedTuple,
    NamedTuple,
} where {T<:half_quant,U<:full_quant}

    CUDA.@fastmath Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(autoMALA_logpos_reduced_4D),
        Enzyme.Active,
        Enzyme.Duplicated(z, ∇z),
        Enzyme.Const(x),
        Enzyme.Const(temps),
        Enzyme.Const(model),
        Enzyme.Const(ps),
        Enzyme.Const(st_kan),
        Enzyme.Const(st_lux),
    )

    logpos, st_ebm, st_gen =
        CUDA.@fastmath autoMALA_logpos_value_4D(z, x, temps, model, ps, st_kan, st_lux)
    return logpos, ∇z, st_ebm, st_gen
end

function autoMALA_logpos(
    z::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant,U<:full_quant}
    st_ebm, st_gen = st_kan.ebm, st_lux.gen
    lp, st_ebm = model.log_prior(z, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)
    ll, st_gen =
        log_likelihood_MALA(z, x, model.lkhood, ps.gen, st_kan.gen, st_lux.gen; ε = model.ε)
    return (lp + temps .* ll) .* model.loss_scaling, st_ebm, st_gen
end

function closure(
    z::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::T where {T<:half_quant,U<:full_quant}
    return sum(first(autoMALA_logpos(z, x, temps, model, ps, st_kan, st_lux)))
end

function autoMALA_value_and_grad(
    z::AbstractArray{T},
    ∇z::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::Tuple{
    AbstractArray{T},
    AbstractArray{T},
    NamedTuple,
    NamedTuple,
} where {T<:half_quant,U<:full_quant}

    CUDA.@fastmath Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(closure),
        Enzyme.Active,
        Enzyme.Duplicated(z, ∇z),
        Enzyme.Const(x),
        Enzyme.Const(temps),
        Enzyme.Const(model),
        Enzyme.Const(ps),
        Enzyme.Const(st_kan),
        Enzyme.Const(st_lux),
    )

    logpos, st_ebm, st_gen =
        CUDA.@fastmath autoMALA_logpos(z, x, temps, model, ps, st_kan, st_lux)
    return logpos, ∇z, st_ebm, st_gen
end

end
