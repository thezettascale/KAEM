module ULA_sampling

export initialize_ULA_sampler, sample

using CUDA,
    KernelAbstractions,
    LinearAlgebra,
    Random,
    Lux,
    LuxCUDA,
    Distributions,
    Accessors,
    Statistics,
    Enzyme,
    ComponentArrays,
    Reactant

include("../../utils.jl")
include("../gen/gen_model.jl")
include("log_posteriors.jl")
using .Utils: device, half_quant, full_quant, fq
using .GeneratorModel: log_likelihood_MALA
using .LogPosteriors: unadjusted_logpos

π_dist = Dict(
    "uniform" => (p, b, rng) -> rand(rng, p, 1, b),
    "gaussian" => (p, b, rng) -> randn(rng, p, 1, b),
    "lognormal" => (p, b, rng) -> rand(rng, LogNormal(0, 1), p, 1, b),
    "ebm" => (p, b, rng) -> randn(rng, p, 1, b),
)

function logpos_grad(
    z_i::AbstractArray{T},
    ∇z::AbstractArray{T},
    x::AbstractArray{T},
    t::AbstractArray{T},
    m::Any,
    ps::ComponentArray{T},
    st::NamedTuple,
    prior_sampling_bool::Bool,
)::AbstractArray{T} where {T<:half_quant}
    CUDA.@fastmath Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        unadjusted_logpos,
        Enzyme.Active,
        Enzyme.Duplicated(T.(z_i), ∇z),
        Enzyme.Const(x),
        Enzyme.Const(t),
        Enzyme.Const(model),
        Enzyme.Const(ps),
        Enzyme.Const(st),
        Enzyme.Const(prior_sampling_bool),
    )
    return ∇z
end

struct ULA_sampler{T<:half_quant,U<:full_quant}
    compiled_llhood::Function
    compiled_logpos_grad::Function
    prior_sampling_bool::Bool
    N::Int
    RE_frequency::Int
    prior_η::U
end


function initialize_ULA_sampler(
    ps::ComponentArray{T},
    st::NamedTuple,
    model::Any,
    x::AbstractArray{T};
    prior_η::U = full_quant(1e-3),
    temps::AbstractArray{T} = [one(half_quant)],
    prior_sampling_bool::Bool = false,
    num_samples::Int = 100,
    N::Int = 20,
    RE_frequency::Int = 10,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:half_quant,U<:full_quant}

    z = begin
        if model.prior.ula && prior_sampling_bool
            z =
                π_dist[model.prior.prior_type](model.prior.p_size, num_samples, rng) |>
                device
        else
            z, st_ebm = model.prior.sample_z(model, size(x)[end]*length(temps), ps, st, rng)
            @reset st.ebm = st_ebm
        end
    end

    num_temps, Q, P, S = length(temps), size(z)[1:2]..., size(x)[end]
    S = prior_sampling_bool ? size(z)[end] : S
    z = reshape(z, Q, P, S, num_temps)
    ∇z = Enzyme.make_zero(z) |> device

    ll =
        (z, x, lkhood, ps_gen, st_gen) ->
            log_likelihood_MALA(z, x, lkhood, ps_gen, st_gen; ε = model.ε)
    compiled_llhood = Reactant.@compile ll(z, x, model.lkhood, ps.gen, st.gen)

    logpos_grad_compiled =
        Reactant.@compile logpos_grad(z, ∇z, x, temps, model, ps, st, prior_sampling_bool)

    return ULA_sampler(
        compiled_llhood,
        logpos_grad_compiled,
        prior_sampling_bool,
        N,
        RE_frequency,
        prior_η,
    )
end


function sample(
    sampler::ULA_sampler,
    model::Any,
    ps::ComponentArray{T},
    st::NamedTuple,
    x::AbstractArray{T};
    temps::AbstractArray{T} = [one(half_quant)],
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
    """
    Unadjusted Langevin Algorithm (ULA) sampler to generate posterior samples.

    Args:
        m: The model.
        ps: The parameters of the model.
        st: The states of the model.
        x: The data.
        t: The temperatures if using Thermodynamic Integration.
        N: The number of iterations.
        rng: The random number generator.

        
    Unused arguments:
        N_unadjusted: The number of unadjusted iterations.
        Δη: The step size increment.
        η_min: The minimum step size.
        η_max: The maximum step size.

    Returns:
        The posterior samples.
    """
    # Initialize from prior
    z = begin
        if model.prior.ula && sampler.prior_sampling_bool
            z =
                full_quant.(
                    π_dist[model.prior.prior_type](model.prior.p_size, num_samples, rng),
                ) |> device
        else
            z, st_ebm = model.prior.sample_z(m, size(x)[end]*length(temps), ps, st, rng)
            @reset st.ebm = st_ebm
            full_quant.(z)
        end
    end

    loss_scaling = model.loss_scaling |> full_quant

    η = sampler.prior_sampling_bool ? sampler.prior_η : mean(st.η_init)
    seq = model.lkhood.seq_length > 1

    num_temps, Q, P, S = length(temps), size(z)[1:2]..., size(x)[end]
    S = sampler.prior_sampling_bool ? size(z)[end] : S
    z = reshape(z, Q, P, S, num_temps)
    ∇z = zeros(T, size(z)) |> device

    # Pre-allocate noise
    noise = randn(rng, full_quant, Q, P, S, num_temps, sampler.N)
    log_u_swap = log.(rand(rng, num_temps, sampler.N)) |> device

    for i = 1:sampler.N
        ξ = device(noise[:, :, :, :, i])
        ∇z =
            full_quant.(
                sampler.compiled_logpos_grad(
                    T.(z),
                    T.(∇z),
                    x,
                    t,
                    model,
                    ps,
                    st,
                    sampler.prior_sampling_bool,
                ),
            ) / loss_scaling
        z += η .* ∇z .+ sqrt(2 * η) .* ξ

        if i % sampler.RE_frequency == 0 && num_temps > 1 && !sampler.prior_sampling_bool
            z_hq = T.(z)
            for t = 1:(num_temps-1)
                ll_t, st_gen = sampler.compiled_llhood(
                    z_hq[:, :, :, t],
                    x,
                    model.lkhood,
                    ps.gen,
                    st_gen;
                )
                ll_t1, st_gen = sampler.compiled_llhood(
                    z_hq[:, :, :, t+1],
                    x,
                    model.lkhood,
                    ps.gen,
                    st_gen;
                )
                log_swap_ratio = dropdims(
                    sum((temps[t+1] - temps[t]) .* (ll_t - ll_t1); dims = 1);
                    dims = 1,
                )
                swap = log_u_swap[t, i] < mean(log_swap_ratio)
                @reset st.gen = st_gen

                # Swap population if likelihood of population in new temperature is higher on average
                if swap
                    z[:, :, :, t] .= z[:, :, :, t+1]
                    z[:, :, :, t+1] .= z[:, :, :, t]
                end
            end
        end
    end

    if sampler.prior_sampling_bool
        st = st.ebm
        z = dropdims(z; dims = 4)
    end

    return T.(z), st
end


end
