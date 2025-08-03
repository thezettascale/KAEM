module LogPosteriors

using CUDA, ComponentArrays, Statistics, Lux, LuxCUDA, LinearAlgebra, Random, Enzyme, Zygote

using ..Utils
using ..T_KAM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

### ULA ### 
function unadjusted_logpos(
    z::AbstractArray{T,3},
    x::AbstractArray{T},
    temps::AbstractArray{T,1},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    prior_sampling_bool::Bool,
    zero_vector::AbstractArray{T},
)::T where {T<:half_quant,U<:full_quant}
    lp = sum(first(model.log_prior(z, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)))
    ll = first(
        log_likelihood_MALA(
            z,
            x,
            model.lkhood,
            ps.gen,
            st_kan.gen,
            st_lux.gen,
            zero_vector;
            ε = model.ε,
        ),
    )
    tempered_ll = sum(temps .* ll)
    return (lp + tempered_ll) * model.loss_scaling
end

function unadjusted_logpos_grad(
    z::AbstractArray{T,3},
    ∇z::AbstractArray{T,3},
    x::AbstractArray{T},
    temps::AbstractArray{T,1},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    prior_sampling_bool::Bool,
)::AbstractArray{T} where {T<:half_quant,U<:full_quant}

    zero_vector = zeros(T, model.lkhood.x_shape..., size(z)[end]) |> pu

    if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
        f =
            z_i -> unadjusted_logpos(
                z_i,
                x,
                temps,
                model,
                ps,
                st_kan,
                st_lux,
                prior_sampling_bool,
                zero_vector,
            )
        ∇z = CUDA.@fastmath first(Zygote.gradient(f, z))
    else
        CUDA.@fastmath Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            Enzyme.Const(unadjusted_logpos),
            Enzyme.Active,
            Enzyme.Duplicated(z, ∇z),
            Enzyme.Const(x),
            Enzyme.Const(temps),
            Enzyme.Const(model),
            Enzyme.Const(ps),
            Enzyme.Const(st_kan),
            Enzyme.Const(st_lux),
            Enzyme.Const(prior_sampling_bool),
            Enzyme.Const(zero_vector),
        )
    end

    all(iszero, ∇z) && error("All zero ULA grad")
    any(isnan, ∇z) && error("NaN ULA grad")
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
    zero_vector::AbstractArray{T},
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant,U<:full_quant}
    Q, P, S, num_temps = size(z)
    z_reshaped = reshape(z, Q, P, S*num_temps)
    x_reshaped = reshape(x, model.lkhood.x_shape..., S*num_temps)
    lp, st_ebm = model.log_prior(z_reshaped, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)
    ll, st_gen = log_likelihood_MALA(
        z_reshaped,
        x_reshaped,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux.gen,
        zero_vector;
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
    zero_vector::AbstractArray{T},
)::T where {T<:half_quant,U<:full_quant}
    Q, P, S, num_temps = size(z)
    z_reshaped = reshape(z, Q, P, S*num_temps)
    x_reshaped = reshape(x, model.lkhood.x_shape..., S*num_temps)
    lp =
        sum(first(model.log_prior(z_reshaped, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)))
    ll = first(
        log_likelihood_MALA(
            z_reshaped,
            x_reshaped,
            model.lkhood,
            ps.gen,
            st_kan.gen,
            st_lux.gen,
            zero_vector;
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

    zero_vector = zeros(T, model.lkhood.x_shape..., prod(size(z)[3:4])) |> pu

    if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
        f =
            z_i -> autoMALA_logpos_reduced_4D(
                z_i,
                x,
                temps,
                model,
                ps,
                st_kan,
                st_lux,
                zero_vector,
            )
        ∇z = CUDA.@fastmath first(Zygote.gradient(f, z))
    else
        CUDA.@fastmath Enzyme.autodiff(
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
            Enzyme.Const(zero_vector),
        )
    end

    logpos, st_ebm, st_gen = CUDA.@fastmath autoMALA_logpos_value_4D(
        z,
        x,
        temps,
        model,
        ps,
        st_kan,
        st_lux,
        zero_vector,
    )
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
    zero_vector::AbstractArray{T},
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant,U<:full_quant}
    st_ebm, st_gen = st_kan.ebm, st_lux.gen
    lp, st_ebm = model.log_prior(z, model.prior, ps.ebm, st_kan.ebm, st_lux.ebm)
    ll, st_gen = log_likelihood_MALA(
        z,
        x,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux.gen,
        zero_vector;
        ε = model.ε,
    )
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
    zero_vector::AbstractArray{T},
)::T where {T<:half_quant,U<:full_quant}
    return sum(first(autoMALA_logpos(z, x, temps, model, ps, st_kan, st_lux, zero_vector)))
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

    zero_vector = zeros(T, model.lkhood.x_shape..., size(z)[end]) |> pu

    if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
        f = z_i -> closure(z_i, x, temps, model, ps, st_kan, st_lux, zero_vector)
        ∇z = CUDA.@fastmath first(Zygote.gradient(f, z))
    else
        CUDA.@fastmath Enzyme.autodiff(
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
            Enzyme.Const(zero_vector),
        )
    end

    logpos, st_ebm, st_gen =
        CUDA.@fastmath autoMALA_logpos(z, x, temps, model, ps, st_kan, st_lux, zero_vector)
    return logpos, ∇z, st_ebm, st_gen
end

end
