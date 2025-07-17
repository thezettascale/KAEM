module LogPosteriors

using CUDA,
    KernelAbstractions, ComponentArrays, Statistics, Lux, LuxCUDA, LinearAlgebra, Random

include("../../utils.jl")
include("../gen/gen_model.jl")
using .Utils: device, half_quant, full_quant
using .GeneratorModel: log_likelihood_MALA

### ULA ### 
function unadjusted_logpos(
    z_i::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    m::Any,
    ps::ComponentArray{T},
    st::NamedTuple,
    prior_sampling_bool::Bool = false,
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    tot = zero(T)
    st_ebm, st_gen = st.ebm, st.gen

    for t_k in temps
        lp, st_ebm = m.prior.lp_fcn(z_i[:, :, :, k], m.prior, ps.ebm, st_ebm; ε = m.ε)
        ll, st_gen =
            log_likelihood_MALA(z_i[:, :, :, k], x, m.lkhood, ps.gen, st_gen; ε = m.ε)
        tot += sum(lp) + (t_k * T(!prior_sampling_bool) * sum(ll))
    end

    return tot * m.loss_scaling
end

### autoMALA ###
function autoMALA_logpos_4D(
    z_i::AbstractArray{T},
    x_i::AbstractArray{T},
    t::AbstractArray{T},
    st_i::NamedTuple,
    m::Any,
    ps::ComponentArray{T},
    num_temps::Int,
    seq::Bool,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    logpos = zeros(T, size(z_i, 3), 0) |> device
    st_ebm, st_gen = st_i.ebm, st_i.gen

    for k = 1:num_temps
        x_k = seq ? x_i[:, :, :, k] : x_i[:, :, :, :, k]
        lp, st_ebm = m.prior.lp_fcn(z_i[:, :, :, k], m.prior, ps.ebm, st_ebm; ε = m.ε)
        ll, st_gen =
            log_likelihood_MALA(z_i[:, :, :, k], x_k, m.lkhood, ps.gen, st_gen; ε = m.ε)
        logpos = hcat(logpos, logprior + t[:, k] .* logllhood)
    end

    return logpos .* m.loss_scaling, st_ebm, st_gen
end

function autoMALA_logpos(
    z_i::AbstractArray{T},
    x_i::AbstractArray{T},
    t::AbstractArray{T},
    st_i::NamedTuple,
    m::Any,
    ps::ComponentArray{T},
    num_temps::Int,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    st_ebm, st_gen = st_i.ebm, st_i.gen
    lp, st_ebm = m.prior.lp_fcn(z_i, m.prior, ps.ebm, st_ebm; ε = m.ε)
    ll, st_gen = log_likelihood_MALA(z_i, x, m.lkhood, ps.gen, st_gen; ε = m.ε)
    return (lp + t .* ll) .* m.loss_scaling, st_ebm, st_gen
end

autoMALA_value_and_grad_4D(
    z_i::AbstractArray{T},
    ∇z::AbstractArray{T},
    x_i::AbstractArray{T},
    t::AbstractArray{T},
    st_i::NamedTuple,
    m::Any,
    ps::ComponentArray{T},
    num_temps::Int,
    seq::Bool,
    )::Tuple{T,AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    fcn = (z, x, temps, s, model, p, n, sequence) -> sum(first(autoMALA_logpos_4D(z, x, temps, s, model, p, n, sequence)))

    CUDA.@fastmath Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        fcn,
        Enzyme.Active,
        Enzyme.Duplicated(T.(z_i), ∇z),
        Enzyme.Const(x_i),
        Enzyme.Const(t),
        Enzyme.Const(Lux.testmode(st_i)),
        Enzyme.Const(m),
        Enzyme.Const(ps),
        Enzyme.Const(num_temps),
        Enzyme.Const(seq),
    )
    
    logpos, st_ebm, st_gen = CUDA.@fastmath autoMALA_logpos_4D(z_i, x_i, t, st_i, m, ps, num_temps, seq)
    return logpos, ∇z, st_ebm, st_gen
end

function autoMALA_value_and_grad(
    z_i::AbstractArray{T},
    ∇z::AbstractArray{T},
    x_i::AbstractArray{T},
    t::AbstractArray{T},
    st_i::NamedTuple,
    m::Any,
    ps::ComponentArray{T},
    num_temps::Int,
    seq::Bool,
    )::Tuple{T,AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    fcn = (z, x, temps, s, model, p, n) -> sum(first(autoMALA_logpos(z, x, temps, s, model, p, n)))

    CUDA.@fastmath Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        fcn,
        Enzyme.Active,
        Enzyme.Duplicated(T.(z_i), ∇z),
        Enzyme.Const(x_i),
        Enzyme.Const(t),
        Enzyme.Const(Lux.testmode(st_i)),
        Enzyme.Const(m),
        Enzyme.Const(ps),
        Enzyme.Const(num_temps),
    )

    logpos, st_ebm, st_gen = CUDA.@fastmath autoMALA_logpos(z_i, x_i, t, st_i, m, ps, num_temps)
    return logpos, ∇z, st_ebm, st_gen
end

end

