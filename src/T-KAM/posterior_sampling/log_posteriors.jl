module LogPosteriors

using CUDA, KernelAbstractions, ComponentArrays, Statistics, Lux, LuxCUDA, LinearAlgebra, Random

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
    st::NamedTuple;
    prior_sampling_bool::Bool = false,
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{T,NamedTuple,NamedTuple} where {T<:half_quant}
    tot = zero(T)
    st_ebm, st_gen = st.ebm, st.gen

    for t_k in temps
        lp, st_ebm = m.prior.lp_fcn(z_i[:, :, :, k], m.prior, ps.ebm, st_ebm; ε = m.ε)
        ll, st_gen = log_likelihood_MALA(
            z_i[:, :, :, k],
            x,
            m.lkhood,
            ps.gen,
            st_gen;
            rng = rng,
            ε = m.ε,
        )
        tot += sum(lp) + (t_k * T(!prior_sampling_bool) * sum(ll))
    end

    return tot * m.loss_scaling, st_ebm, st_gen
end

### autoMALA ###
function autoMALA_logpos_4D(
    z_i::AbstractArray{T},
    x_i::AbstractArray{T},
    t::AbstractArray{T},
    st_i::NamedTuple,
    m::Any,
    ps::ComponentArray{T};
    rng::AbstractRNG = Random.default_rng(),
    num_temps::Int = 1,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    logpos = zeros(T, size(z_i, 3), 0) |> device
    st_ebm, st_gen = st_i.ebm, st_i.gen

    for k = 1:num_temps
        x_k = seq ? x_i[:, :, :, k] : x_i[:, :, :, :, k]
        lp, st_ebm = m.prior.lp_fcn(z_i[:, :, :, k], m.prior, ps.ebm, st_ebm; ε = m.ε)
        ll, st_gen = log_likelihood_MALA(
            z_i[:, :, :, k],
            x_k,
            m.lkhood,
            ps.gen,
            st_gen;
            rng = rng,
            ε = m.ε,
        )
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
    ps::ComponentArray{T};
    rng::AbstractRNG = Random.default_rng(),
    num_temps::Int = 1,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple} where {T<:half_quant}
    st_ebm, st_gen = st_i.ebm, st_i.gen
    lp, st_ebm = m.prior.lp_fcn(z_i, m.prior, ps.ebm, st_ebm; ε = m.ε)
    ll, st_gen =
        log_likelihood_MALA(z_i, x, m.lkhood, ps.gen, st_gen; rng = rng, ε = m.ε)
    return (lp + t .* ll) .* m.loss_scaling, st_ebm, st_gen
end

end
