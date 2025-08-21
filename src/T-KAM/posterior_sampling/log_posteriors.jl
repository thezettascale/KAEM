module LogPosteriors

using CUDA, ComponentArrays, Statistics, Lux, LuxCUDA, LinearAlgebra, Random, Zygote

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
    return (lp + tempered_ll) * model.loss_scaling.reduced
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

    return CUDA.@fastmath first(Zygote.gradient(f, z))
end

### autoMALA ###
function autoMALA_logpos(
    z::AbstractArray{T,3},
    x::AbstractArray{T},
    temps::AbstractArray{T,1},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    zero_vector::AbstractArray{T},
)::Tuple{AbstractArray{T,1},NamedTuple,NamedTuple} where {T<:half_quant,U<:full_quant}
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
    return (lp + temps .* ll) .* model.loss_scaling.reduced, st_ebm, st_gen
end

function closure(
    z::AbstractArray{T,3},
    x::AbstractArray{T},
    temps::AbstractArray{T,1},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    zero_vector::AbstractArray{T},
)::T where {T<:half_quant,U<:full_quant}
    return sum(first(autoMALA_logpos(z, x, temps, model, ps, st_kan, st_lux, zero_vector)))
end

function autoMALA_value_and_grad(
    z::AbstractArray{T,3},
    ∇z::AbstractArray{T,3},
    x::AbstractArray{T},
    temps::AbstractArray{T,1},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
)::Tuple{
    AbstractArray{T,1},
    AbstractArray{T,3},
    NamedTuple,
    NamedTuple,
} where {T<:half_quant,U<:full_quant}

    zero_vector = zeros(T, model.lkhood.x_shape..., size(z)[end]) |> pu

    f = z_i -> closure(z_i, x, temps, model, ps, st_kan, st_lux, zero_vector)
    ∇z = CUDA.@fastmath first(Zygote.gradient(f, z))

    logpos, st_ebm, st_gen =
        CUDA.@fastmath autoMALA_logpos(z, x, temps, model, ps, st_kan, st_lux, zero_vector)
    return logpos, ∇z, st_ebm, st_gen
end

end
