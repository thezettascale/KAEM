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
using .Utils: device, half_quant, full_quant
using .GeneratorModel: log_likelihood_MALA

### ULA ### 
function unadjusted_logpos(
    z_i::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    m,
    ps::ComponentArray{T},
    st::NamedTuple,
    prior_sampling_bool::Bool,
)::T where {T<:half_quant}
    Q, P, S, num_temps = size(z_i)
    z_reshaped = reshape(z_i, Q, P, S*num_temps)
    lp = sum(first(m.prior.lp_fcn(z_reshaped, m.prior, ps.ebm, st.ebm; ε = m.ε)))
    ll = first(log_likelihood_MALA(z_reshaped, x, m.lkhood, ps.gen, st.gen; ε = m.ε))
    tempered_ll = sum(temps .* reshape(ll, num_temps, S))
    return (lp + tempered_ll) * m.loss_scaling
end

### autoMALA ###
function autoMALA_logpos_value_4D(
    z_i::AbstractArray{T},
    x_i::AbstractArray{T},
    t::AbstractArray{T},
    m,
    ps::ComponentArray{T},
    st_i::NamedTuple,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    Q, P, S, num_temps = size(z_i)
    z_reshaped = reshape(z_i, Q, P, S*num_temps)
    lp, st_ebm = m.prior.lp_fcn(z_reshaped, m.prior, ps.ebm, st_i.ebm; ε = m.ε)
    ll, st_gen = log_likelihood_MALA(z_reshaped, x_i, m.lkhood, ps.gen, st_i.gen; ε = m.ε)
    logpos = reshape(lp, S, num_temps) + t .* reshape(ll, S, num_temps)
    return logpos .* m.loss_scaling, st_ebm, st_gen
end

function autoMALA_logpos_reduced_4D(
    z_i::AbstractArray{T},
    x_i::AbstractArray{T},
    t::AbstractArray{T},
    m,
    ps::ComponentArray{T},
    st_i::NamedTuple,
)::T where {T<:half_quant}
    Q, P, S, num_temps = size(z_i)
    z_reshaped = reshape(z_i, Q, P, S*num_temps)
    lp = sum(first(m.prior.lp_fcn(z_reshaped, m.prior, ps.ebm, st_i.ebm; ε = m.ε)))
    ll = first(log_likelihood_MALA(z_reshaped, x_i, m.lkhood, ps.gen, st_i.gen; ε = m.ε))
    tempered_ll = sum(t .* reshape(ll, S, num_temps))
    return (lp + tempered_ll) * m.loss_scaling
end

function autoMALA_value_and_grad_4D(
    z_i::AbstractArray{T},
    ∇z::AbstractArray{T},
    x_i::AbstractArray{T},
    t::AbstractArray{T},
    m,
    ps::ComponentArray{T},
    st_i::NamedTuple,
    num_temps::Int,
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    x_expanded =
        ndims(x_i) == 4 ? repeat(x_i, 1, 1, 1, num_temps) : repeat(x_i, 1, 1, num_temps)

    CUDA.@fastmath Enzyme.autodiff(
        Enzyme.Reverse,
        autoMALA_logpos_reduced_4D,
        Enzyme.Active,
        Enzyme.Duplicated(z_i, ∇z),
        Enzyme.Const(x_expanded),
        Enzyme.Const(t),
        Enzyme.Const(m),
        Enzyme.Const(ps),
        Enzyme.Const(st_i),
    )

    any(isnan, ∇z) && error("∇z is NaN")
    all(iszero, ∇z) && error("∇z is zero")

    logpos, st_ebm, st_gen =
        CUDA.@fastmath autoMALA_logpos_value_4D(z_i, x_i, t, m, ps, st_i)
    return logpos, ∇z, st_ebm, st_gen
end

function autoMALA_logpos(
    z_i::AbstractArray{T},
    x_i::AbstractArray{T},
    t::AbstractArray{T},
    m,
    ps::ComponentArray{T},
    st_i::NamedTuple,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    st_ebm, st_gen = st_i.ebm, st_i.gen
    lp, st_ebm = m.prior.lp_fcn(z_i, m.prior, ps.ebm, st_ebm; ε = m.ε)
    ll, st_gen = log_likelihood_MALA(z_i, x_i, m.lkhood, ps.gen, st_gen; ε = m.ε)
    return (lp + t .* ll) .* m.loss_scaling, st_ebm, st_gen
end

function autoMALA_value_and_grad(
    z_i::AbstractArray{T},
    ∇z::AbstractArray{T},
    x_i::AbstractArray{T},
    t::AbstractArray{T},
    m,
    ps::ComponentArray{T},
    st_i::NamedTuple,
    num_temps::Int,
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    fcn =
        (z, x, temps, model, p, s) -> sum(first(autoMALA_logpos(z, x, temps, model, p, s)))

    x_expanded =
        ndims(x_i) == 4 ? repeat(x_i, 1, 1, 1, num_temps) : repeat(x_i, 1, 1, num_temps)

    CUDA.@fastmath Enzyme.autodiff(
        Enzyme.Reverse,
        fcn,
        Enzyme.Active,
        Enzyme.Duplicated(z_i, ∇z),
        Enzyme.Const(x_expanded),
        Enzyme.Const(t),
        Enzyme.Const(m),
        Enzyme.Const(ps),
        Enzyme.Const(st_i),
    )

    any(isnan, ∇z) && error("∇z is NaN")
    all(iszero, ∇z) && error("∇z is zero")

    logpos, st_ebm, st_gen = CUDA.@fastmath autoMALA_logpos(z_i, x_i, t, m, ps, st_i)
    return logpos, ∇z, st_ebm, st_gen
end

end
