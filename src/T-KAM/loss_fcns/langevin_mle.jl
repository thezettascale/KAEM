module LangevinMLE

export langevin_loss

using CUDA, KernelAbstractions, Enzyme, ComponentArrays
using Statistics, Lux, LuxCUDA

include("../../utils.jl")
using .Utils: device, next_rng, half_quant, full_quant, hq

function sample_langevin(
    ps::ComponentArray{T},
    st::NamedTuple,
    m::Any,
    x::AbstractArray{T};
    seed::Int = 1,
)::Tuple{AbstractArray{T},NamedTuple,Int} where {T<:half_quant}
    z, st_ebm, seed = m.posterior_sample(m, x, 0, ps, st, seed)
    return z, st_ebm, seed
end

function marginal_llhood(
    ps::ComponentArray{T},
    z::AbstractArray{T},
    x::AbstractArray{T},
    m::Any,
    st_ebm::NamedTuple,
    st_gen::NamedTuple;
    seed::Int = 1,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple,Int} where {T<:half_quant}

    logprior_pos, st_ebm = m.prior.lp_fcn(
        z[:, :, :, 1],
        m.prior,
        ps.ebm,
        st_ebm;
        ε = m.ε,
        normalize = !m.prior.contrastive_div,
    )
    logllhood, st_gen, seed = log_likelihood_MALA(
        z[:, :, :, 1],
        x,
        m.lkhood,
        ps.gen,
        st_gen;
        seed = seed,
        ε = m.ε,
    )
    contrastive_div = mean(logprior_pos)

    if m.prior.contrastive_div
        z, st_ebm, seed = m.prior.sample_z(m, size(x)[end], ps, st, seed)
        logprior, st_ebm = m.prior.lp_fcn(
            z,
            m.prior,
            ps.ebm,
            st_ebm;
            ε = m.ε,
            normalize = !m.prior.contrastive_div,
        )
        contrastive_div -= mean(logprior)
    end

    return -(contrastive_div + mean(logllhood))*m.loss_scaling, st_ebm, st_gen, seed
end

function langevin_loss(
    ps::ComponentArray{T},
    ∇::ComponentArray{T},
    st::NamedTuple,
    model::Any,
    x::AbstractArray{T};
    seed::Int = 1,
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple,Int} where {T<:half_quant}
    """MLE loss without importance, (used when posterior expectation = MCMC estimate)."""

    z, st_new, seed = sample_langevin(ps, st, model, x; seed = seed)
    st_ebm, st_gen = st_new.ebm, st_new.gen

    f =
        (p, z_i, x_i, m, se, sg) -> begin
            first(marginal_llhood(p, z_i, x_i, m, se, sg; seed = seed))
        end

    CUDA.@fastmath Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        f,
        Enzyme.Active,
        Enzyme.Duplicated(ps, ∇),
        Enzyme.Const(z),
        Enzyme.Const(x),
        Enzyme.Const(model),
        Enzyme.Const(st_ebm),
        Enzyme.Const(st_gen),
    )

    loss, st_ebm, st_gen, seed =
        marginal_llhood(ps, z, x, model, st_ebm, st_gen; seed = seed)
    return loss, ∇, st_ebm, st_gen, seed
end

end
