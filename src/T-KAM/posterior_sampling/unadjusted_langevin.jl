module ULA_sampling

export initialize_ULA_sampler, ULA_sampler

using CUDA,
    LinearAlgebra,
    Random,
    Lux,
    LuxCUDA,
    Distributions,
    Accessors,
    Statistics,
    Enzyme,
    ComponentArrays,
    ParallelStencil

using ..Utils
using ..T_KAM_model

include("log_posteriors.jl")
using .LogPosteriors: unadjusted_logpos_grad, log_likelihood_MALA

π_dist = Dict(
    "uniform" => (p, b, rng) -> rand(rng, p, 1, b),
    "gaussian" => (p, b, rng) -> randn(rng, p, 1, b),
    "lognormal" => (p, b, rng) -> rand(rng, LogNormal(0, 1), p, 1, b),
    "ebm" => (p, b, rng) -> randn(rng, p, 1, b),
)

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

@parallel_indices (q, p, s) function update_z!(
    z::AbstractArray{U},
    ∇z::AbstractArray{U},
    η::U,
    ξ::AbstractArray{U},
    sqrt_2η::U,
)::Nothing where {U<:full_quant}
    z[q, p, s] += η * ∇z[q, p, s] + sqrt_2η * ξ[q, p, s]
    return nothing
end

struct ULA_sampler{U<:full_quant}
    prior_sampling_bool::Bool
    N::Int
    RE_frequency::Int
    η::U
end

function initialize_ULA_sampler(;
    η::U = full_quant(1e-3),
    prior_sampling_bool::Bool = false,
    N::Int = 20,
    RE_frequency::Int = 10,
) where {U<:full_quant}

    return ULA_sampler(prior_sampling_bool, N, RE_frequency, η)
end

function (sampler::ULA_sampler)(
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    x::AbstractArray{T};
    temps::AbstractArray{T} = [one(T)],
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant,U<:full_quant}
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
    z_hq = begin
        if model.prior.ula && sampler.prior_sampling_bool
            z = π_dist[model.prior.prior_type](model.prior.p_size, size(x)[end], rng)
            z = pu(z)
        else
            z, st_ebm = model.sample_prior(
                model,
                size(x)[end]*length(temps),
                ps,
                st_kan,
                st_lux,
                rng,
            )
            @reset st_lux.ebm = st_ebm
            z
        end
    end

    loss_scaling = U(model.loss_scaling)

    η = sampler.η
    sqrt_2η = sqrt(2 * η)
    seq = model.lkhood.SEQ

    num_temps, Q, P, S = length(temps), size(z_hq)[1:2]..., size(x)[end]
    S = sampler.prior_sampling_bool ? size(z_hq)[end] : S
    z_hq = reshape(z_hq, Q, P, S, num_temps)
    temps_gpu = pu(repeat(temps, S))

    # Pre-allocate for both precisions
    z_fq = U.(reshape(z_hq, Q, P, S*num_temps))
    ∇z_fq = Enzyme.make_zero(z_fq)
    z_copy = similar(z_hq[:, :, :, 1]) |> pu
    z_t, z_t1 = z_copy, z_copy

    x_t =
        model.lkhood.SEQ ? repeat(x, 1, 1, num_temps) : repeat(x, 1, 1, 1, num_temps)
    
    # Pre-allocate noise
    noise = randn(rng, U, Q, P, S*num_temps, sampler.N)
    log_u_swap = log.(rand(rng, num_temps-1, sampler.N)) |> pu
    ll_noise = randn(rng, T, model.lkhood.x_shape..., S, 2, sampler.N) |> pu

    for i = 1:sampler.N
        ξ = pu(noise[:, :, :, i])
        ∇z_fq .=
            U.(
                unadjusted_logpos_grad(
                    T.(z_fq),
                    Enzyme.make_zero(T.(z_fq)),
                    x_t,
                    temps_gpu,
                    model,
                    ps,
                    st_kan,
                    st_lux,
                    sampler.prior_sampling_bool,
                ),
            ) ./ loss_scaling

        @parallel (1:Q, 1:P, 1:S) update_z!(z_fq, ∇z_fq, η, ξ, sqrt_2η)
        z_hq .= T.(reshape(z_fq, Q, P, S, num_temps))

        if i % sampler.RE_frequency == 0 && num_temps > 1 && !sampler.prior_sampling_bool
            for t = 1:(num_temps-1)

                z_t = copy(z_hq[:, :, :, t])
                z_t1 = copy(z_hq[:, :, :, t+1])

                ll_t, st_gen = log_likelihood_MALA(
                    z_t,
                    x,
                    model.lkhood,
                    ps.gen,
                    st_kan.gen,
                    st_lux.gen,
                    ll_noise[:, :, :, 1, i];
                    ε = model.ε,
                )
                ll_t1, st_gen = log_likelihood_MALA(
                    z_t1,
                    x,
                    model.lkhood,
                    ps.gen,
                    st_kan.gen,
                    st_lux.gen,
                    ll_noise[:, :, :, 2, i];
                    ε = model.ε,
                )

                log_swap_ratio = dropdims(
                    sum((temps[t+1] - temps[t]) .* (ll_t - ll_t1); dims = 1);
                    dims = 1,
                )
                swap = T(log_u_swap[t, i] < mean(log_swap_ratio))
                @reset st_lux.gen = st_gen

                # Swap population if likelihood of population in new temperature is higher on average
                z_hq[:, :, :, t] .= swap .* z_t1 .+ (1 - swap) .* z_t
                z_hq[:, :, :, t+1] .= (1 - swap) .* z_t1 .+ swap .* z_t
                z_fq .= U.(reshape(z_hq, Q, P, S*num_temps))
            end
        end
    end

    if sampler.prior_sampling_bool
        st_lux = st_lux.ebm
        z_hq = dropdims(z_hq; dims = 4)
    end

    return z_hq, st_lux
end


end
