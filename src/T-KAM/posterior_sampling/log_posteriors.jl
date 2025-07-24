module LogPosteriors

using CUDA,
    KernelAbstractions,
    ComponentArrays,
    Statistics,
    Lux,
    LuxCUDA,
    LinearAlgebra,
    Random,
    Enzyme

include("../../utils.jl")
include("../gen/gen_model.jl")
include("../T-KAM.jl")
using .Utils: device, half_quant, full_quant
using .GeneratorModel: log_likelihood_MALA
using .T_KAM: T_KAM

### ULA ### 
function unadjusted_logpos(
    z::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    model::T_KAM{T,full_quant},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    prior_sampling_bool::Bool,
)::T where {T<:half_quant}
    Q, P, S, num_temps = size(z)
    z_reshaped = reshape(z, Q, P, S*num_temps)
    lp = sum(
        first(
            model.prior.lp_fcn(
                z_reshaped,
                model.prior,
                ps.ebm,
                st_kan.ebm,
                st_lux.ebm;
                ε = model.ε,
            ),
        ),
    )
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
    model::T_KAM{T,full_quant},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    prior_sampling_bool::Bool,
)::AbstractArray{T} where {T<:half_quant}

    # Expand for log_likelihood
    x_expanded =
        ndims(x) == 4 ? repeat(x, 1, 1, 1, length(temps)) : repeat(x, 1, 1, length(temps))

    CUDA.@fastmath Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        unadjusted_logpos,
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

    # any(isnan, ∇z) && error("∇z is NaN")
    # all(iszero, ∇z) && error("∇z is zero")
    return ∇z
end

### autoMALA ###
function autoMALA_logpos_value_4D(
    z::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    model::T_KAM{T,full_quant},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    Q, P, S, num_temps = size(z)
    z_reshaped = reshape(z, Q, P, S*num_temps)
    lp, st_ebm = model.prior.lp_fcn(
        z_reshaped,
        model.prior,
        ps.ebm,
        st_kan.ebm,
        st_lux.ebm;
        ε = model.ε,
    )
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
    model::T_KAM{T,full_quant},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::T where {T<:half_quant}
    Q, P, S, num_temps = size(z)
    z_reshaped = reshape(z, Q, P, S*num_temps)
    lp = sum(
        first(
            model.prior.lp_fcn(
                z_reshaped,
                model.prior,
                ps.ebm,
                st_kan.ebm,
                st_lux.ebm;
                ε = model.ε,
            ),
        ),
    )
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
    model::T_KAM{T,full_quant},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    x_expanded =
        ndims(x) == 4 ? repeat(x, 1, 1, 1, length(temps)) : repeat(x, 1, 1, length(temps))

    CUDA.@fastmath Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        autoMALA_logpos_reduced_4D,
        Enzyme.Active,
        Enzyme.Duplicated(z, ∇z),
        Enzyme.Const(x_expanded),
        Enzyme.Const(temps),
        Enzyme.Const(model),
        Enzyme.Const(ps),
        Enzyme.Const(st_kan),
        Enzyme.Const(st_lux),
    )

    any(isnan, ∇z) && error("∇z is NaN")
    all(iszero, ∇z) && error("∇z is zero")

    logpos, st_ebm, st_gen =
        CUDA.@fastmath autoMALA_logpos_value_4D(z, x, temps, m, ps, st_kan, st_lux)
    return logpos, ∇z, st_ebm, st_gen
end

function autoMALA_logpos(
    z::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    model::T_KAM{T,full_quant},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    st_ebm, st_gen = st_kan.ebm, st_lux.gen
    lp, st_ebm =
        model.prior.lp_fcn(z, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm; ε = model.ε)
    ll, st_gen =
        log_likelihood_MALA(z, x, model.lkhood, ps.gen, st_kan.gen, st_lux.gen; ε = model.ε)
    return (lp + temps .* ll) .* model.loss_scaling, st_ebm, st_gen
end

function closure(
    z::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    model::T_KAM{T,full_quant},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::T where {T<:half_quant}
    return sum(first(autoMALA_logpos(z, x, temps, model, ps, st_kan, st_lux)))
end

function autoMALA_value_and_grad(
    z::AbstractArray{T},
    ∇z::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    model::T_KAM{T,full_quant},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    x_expanded =
        ndims(x) == 4 ? repeat(x, 1, 1, 1, length(temps)-size(x)[end]) :
        repeat(x, 1, 1, length(temps)-size(x)[end])

    CUDA.@fastmath Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        closure,
        Enzyme.Active,
        Enzyme.Duplicated(z, ∇z),
        Enzyme.Const(x_expanded),
        Enzyme.Const(temps),
        Enzyme.Const(model),
        Enzyme.Const(ps),
        Enzyme.Const(st_kan),
        Enzyme.Const(st_lux),
    )

    any(isnan, ∇z) && error("∇z is NaN")
    all(iszero, ∇z) && error("∇z is zero")

    logpos, st_ebm, st_gen =
        CUDA.@fastmath autoMALA_logpos(z, x, temps, model, ps, st_kan, st_lux)
    return logpos, ∇z, st_ebm, st_gen
end

end
