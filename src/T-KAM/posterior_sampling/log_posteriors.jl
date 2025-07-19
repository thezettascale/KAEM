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
    prior_sampling_bool::Bool = false,
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    tot = zero(T)
    st_ebm, st_gen = st.ebm, st.gen

    for k in eachindex(temps)
        lp, st_ebm = m.prior.lp_fcn(z_i[:, :, :, k], m.prior, ps.ebm, st_ebm; ε = m.ε)
        tot += sum(lp) 
        if !prior_sampling_bool
            ll, st_gen =
                log_likelihood_MALA(z_i[:, :, :, k], x, m.lkhood, ps.gen, st_gen; ε = m.ε)
            tot += sum(temps[k] .* ll)
        end
    end

    return tot * m.loss_scaling
end

### autoMALA ###
function autoMALA_logpos_4D(
    z_i::AbstractArray{T},
    x_i::AbstractArray{T},
    t::AbstractArray{T},
    m,
    ps::ComponentArray{T},
    st_i::NamedTuple,
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
        logpos = hcat(logpos, lp + t[:, k] .* ll)
    end

    return logpos .* m.loss_scaling, st_ebm, st_gen
end

function autoMALA_logpos(
    z_i::AbstractArray{T},
    x_i::AbstractArray{T},
    t::AbstractArray{T},
    m,
    ps::ComponentArray{T},
    st_i::NamedTuple,
    num_temps::Int,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    st_ebm, st_gen = st_i.ebm, st_i.gen
    lp, st_ebm = m.prior.lp_fcn(z_i, m.prior, ps.ebm, st_ebm; ε = m.ε)
    ll, st_gen = log_likelihood_MALA(z_i, x_i, m.lkhood, ps.gen, st_gen; ε = m.ε)
    return (lp + t .* ll) .* m.loss_scaling, st_ebm, st_gen
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
    seq::Bool,
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    fcn =
        (z, x, temps, model, p, s, n, sequence) ->
            sum(first(autoMALA_logpos_4D(z, x, temps, model, p, s, n, sequence)))

    CUDA.@fastmath Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        fcn,
        Enzyme.Active,
        Enzyme.Duplicated(T.(z_i), ∇z),
        Enzyme.Const(x_i),
        Enzyme.Const(t),
        Enzyme.Const(m),
        Enzyme.Const(ps),
        Enzyme.Const(Lux.testmode(st_i)),
        Enzyme.Const(num_temps),
        Enzyme.Const(seq),
    )

    logpos, st_ebm, st_gen =
        CUDA.@fastmath autoMALA_logpos_4D(z_i, x_i, t, m, ps, st_i, num_temps, seq)
    return logpos, ∇z, st_ebm, st_gen
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
    seq::Bool,
)::Tuple{AbstractArray{T},AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}

    fcn =
        (z, x, temps, model, p, s, n) ->
            sum(first(autoMALA_logpos(z, x, temps, model, p, s, n)))

    CUDA.@fastmath Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        fcn,
        Enzyme.Active,
        Enzyme.Duplicated(T.(z_i), ∇z),
        Enzyme.Const(x_i),
        Enzyme.Const(t),
        Enzyme.Const(m),
        Enzyme.Const(ps),
        Enzyme.Const(Lux.testmode(st_i)),
        Enzyme.Const(num_temps),
    )

    logpos, st_ebm, st_gen =
        CUDA.@fastmath autoMALA_logpos(z_i, x_i, t, m, ps, st_i, num_temps)
    return logpos, ∇z, st_ebm, st_gen
end

end
