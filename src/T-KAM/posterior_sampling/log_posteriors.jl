module LogPosteriors


### ULA ### 
function unadjusted_logpos(
    z_i::AbstractArray{T},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    m::Any,
    ps::ComponentArray{T},
    st::NamedTuple,
    prior_sampling_bool::Bool = false,
    seed::Int = 1,
)::Tuple{T,NamedTuple,NamedTuple,Int} where {T<:half_quant}
    tot = zero(T)
    st_ebm, st_gen = st.ebm, st.gen

    for t_k in temps
        lp, st_ebm = m.prior.lp_fcn(z_i[:, :, :, k], m.prior, ps.ebm, st_ebm; ε = m.ε)
        ll, st_gen, seed = log_likelihood_MALA(
            z_i[:, :, :, k],
            x,
            m.lkhood,
            ps.gen,
            st_gen;
            seed = seed,
            ε = m.ε,
        )
        tot += sum(lp) + (t_k * T(!prior_sampling_bool) * sum(ll))
    end

    return tot * m.loss_scaling, st_ebm, st_gen, seed
end

### autoMALA ###
function autoMALA_logpos_4D(
    z_i::AbstractArray{T},
    x_i::AbstractArray{T},
    t::AbstractArray{T},
    st_i::NamedTuple,
    m::Any,
    ps::ComponentArray{T};
    seed::Int = 1,
    num_temps::Int = 1,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple,Int} where {T<:half_quant}
    logpos = zeros(T, S, 0) |> device
    st_ebm, st_gen = st_i.ebm, st_i.gen

    for k = 1:num_temps
        x_k = seq ? x_i[:, :, :, k] : x_i[:, :, :, :, k]
        lp, st_ebm = m.prior.lp_fcn(z_i[:, :, :, k], m.prior, ps.ebm, st_ebm; ε = m.ε)
        ll, st_gen, seed = log_likelihood_MALA(
            z_i[:, :, :, k],
            x_k,
            m.lkhood,
            ps.gen,
            st_gen;
            seed = seed,
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
    seed::Int = 1,
    num_temps::Int = 1,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple,Int} where {T<:half_quant}
    st_ebm, st_gen = st_i.ebm, st_i.gen
    lp, st_ebm = m.prior.lp_fcn(z_i, m.prior, ps.ebm, st_ebm; ε = m.ε)
    ll, st_gen, seed =
        log_likelihood_MALA(z_i, x, m.lkhood, ps.gen, st_gen; seed = seed, ε = m.ε)
    return (lp + t .* ll) .* m.loss_scaling, st_ebm, st_gen, seed
end

end
