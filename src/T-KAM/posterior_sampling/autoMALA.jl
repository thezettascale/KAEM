module autoMALA_sampling

export initialize_autoMALA_sampler, autoMALA_sampler

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

include("preconditioner.jl")
using .Preconditioning

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

if parse(Bool, get(ENV, "THERMO", "false"))
    include("thermo_updates.jl")
    using .LangevinUpdates
else
    include("updates.jl")
    using .LangevinUpdates
end

include("step_search.jl")
using .autoMALA_StepSearch

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

const target_rate = 0.574  # Optimal acceptance rate for MALA

struct autoMALA_sampler{U<:full_quant}
    N::Int
    N_unadjusted::Int
    η::AbstractArray{U}
    Δη::U
    η_min::U
    η_max::U
    RE_frequency::Int
end

function initialize_autoMALA_sampler(;
    N::Int = 20,
    N_unadjusted::Int = 1,
    RE_frequency::Int = 10,
    η::U = full_quant(1e-3),
    Δη::U = full_quant(2),
    η_min::U = full_quant(1e-5),
    η_max::U = one(full_quant),
    samples::Int = 100,
    num_temps::Int = 1,
) where {U<:full_quant}

    return autoMALA_sampler(
        N,
        N_unadjusted,
        repeat([η], samples*num_temps) |> pu,
        Δη,
        η_min,
        η_max,
        RE_frequency,
    )
end

@parallel_indices (q, p, s) function accept_reject!(
    z_fq::AbstractArray{U,3},
    log_u::AbstractArray{U,1},
    log_r::AbstractArray{U,1},
    reversible::AbstractArray{Bool,1},
    mean_η::AbstractArray{U,1},
    η::AbstractArray{U,1},
    η_prop::AbstractArray{U,1},
    num_acceptances::AbstractArray{Int,1},
    ẑ::AbstractArray{U,3},
)::Nothing where {U<:full_quant}
    accept = (log_u[s] < log_r[s]) * reversible[s]
    z_fq[q, p, s] = ẑ[q, p, s] * accept + z_fq[q, p, s] * (1 - accept)
    mean_η[s] = mean_η[s] + η_prop[s] * accept
    η[s] = η_prop[s] * accept + η[s] * (1 - accept)
    num_acceptances[s] = num_acceptances[s] + accept
    return nothing
end

function (sampler::autoMALA_sampler)(
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    x::AbstractArray{T};
    temps::AbstractArray{T} = [one(T)],
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant,U<:full_quant}
    """
    Metropolis-adjusted Langevin algorithm (MALA) sampler to generate posterior samples.

    Args:
        m: The model.
        ps: The parameters of the model.
        st: The states of the model.
        x: The data.
        t: The temperatures if using Thermodynamic Integration.
        N: The number of iterations.
        rng: The random number generator.
    """
    # Initialize from prior 
    z_hq, st_ebm = model.sample_prior(
        model,
        size(x)[end],
        ps,
        st_kan,
        st_lux,
        rng
    )
    for i in 1:length(temps)-1
        z_i, st_ebm = model.sample_prior(model, size(x)[end], ps, st_kan, st_lux, rng)
        z_hq = cat(z_hq, z_i; dims = 3)
    end
    @reset st_lux.ebm = st_ebm

    num_temps, Q, P, S = length(temps), size(z_hq)[1:2]..., size(x)[end]
    z_hq = reshape(z_hq, Q, P, S, num_temps)
    ∇z_hq = Enzyme.make_zero(z_hq)

    # Pre-allocate for both precisions
    z_fq = U.(reshape(z_hq, Q, P, S*num_temps))
    ∇z_fq = Enzyme.make_zero(z_fq)
    z_copy = similar(z_hq[:, :, :, 1]) |> pu
    z_t, z_t1 = z_copy, z_copy

    t_expanded = repeat(temps, S) |> pu
    x_t = model.lkhood.SEQ ? repeat(x, 1, 1, num_temps) : repeat(x, 1, 1, 1, num_temps)

    # Initialize preconditioner
    M = init_mass_matrix(z_fq)
    @reset sampler.η = pu(sampler.η)

    log_u = log.(rand(rng, S*num_temps, sampler.N)) |> pu
    ratio_bounds = log.(U.(rand(rng, Uniform(0, 1), S*num_temps, 2, sampler.N))) |> pu
    log_u_swap = log.(rand(rng, U, S, num_temps-1, sampler.N))
    ll_noise = randn(rng, T, model.lkhood.x_shape..., S, 2, num_temps, sampler.N) |> pu
    swap_replica_idxs = num_temps > 1 ? rand(rng, 1:num_temps-1, sampler.N) : nothing

    num_acceptances = zeros(Int, S*num_temps) |> pu
    mean_η = zeros(U, S*num_temps) |> pu
    momentum = Enzyme.make_zero(z_fq)

    burn_in = 0
    η = sampler.η

    for i = 1:sampler.N
        momentum, M = sample_momentum(z_fq, M)

        log_a, log_b = dropdims(minimum(ratio_bounds[:, :, i]; dims = 2); dims = 2),
        dropdims(maximum(ratio_bounds[:, :, i]; dims = 2); dims = 2)

        logpos_z, ∇z_fq, st_lux =
            logpos_withgrad(T.(z_fq), T.(∇z_fq), x_t, t_expanded, model, ps, st_kan, st_lux)

        if burn_in < sampler.N
            burn_in += 1
            z_fq, logpos_ẑ, ∇ẑ, p̂, log_r, st_lux = leapfrog(
                z_fq,
                ∇z_fq,
                x_t,
                t_expanded,
                logpos_z,
                momentum,
                M,
                η,
                model,
                ps,
                st_kan,
                st_lux,
            )
            z_hq .= T.(reshape(z_fq, Q, P, S, num_temps))

        else
            ẑ, η_prop, η_prime, reversible, log_r, st_lux = autoMALA_step(
                log_a,
                log_b,
                z_fq,
                ∇z_fq,
                x_t,
                t_expanded,
                logpos_z,
                momentum,
                M,
                model,
                ps,
                st_kan,
                st_lux,
                η,
                sampler.Δη,
                sampler.η_min,
                sampler.η_max,
                model.ε,
            )

            @parallel (1:Q, 1:P, 1:(S*num_temps)) accept_reject!(
                z_fq,
                log_u[:, :, i],
                log_r,
                reversible,
                mean_η,
                η,
                η_prop,
                num_acceptances,
                ẑ,
            )

            z_hq .= T.(reshape(z_fq, Q, P, S, num_temps))

            # Replica exchange Monte Carlo
            if i % sampler.RE_frequency == 0 && num_temps > 1
                t = swap_replica_idxs[i] # Randomly pick two adjacent temperatures to swap
                z_t = copy(z_hq[:, :, :, t])
                z_t1 = copy(z_hq[:, :, :, t+1])

                noise_1 = model.lkhood.SEQ ? ll_noise[:, :, :, 1, t, i] : ll_noise[:, :, :, :, 1, t, i]
                noise_2 = model.lkhood.SEQ ? ll_noise[:, :, :, 2, t, i] : ll_noise[:, :, :, :, 2, t, i]

                ll_t, st_gen = log_likelihood_MALA(
                    z_t,
                    x,
                    model.lkhood,
                    ps.gen,
                    st_kan.gen,
                    st_lux.gen,
                    noise_1;
                    ε = model.ε,
                )
                ll_t1, st_gen = log_likelihood_MALA(
                    z_t1,
                    x,
                    model.lkhood,
                    ps.gen,
                    st_kan.gen,
                    st_gen,
                    noise_2;
                    ε = model.ε,
                )

                log_swap_ratio = (temps[t+1] - temps[t]) .* (sum(ll_t) - sum(ll_t1))
                swap = T(log_u_swap[t, i] < log_swap_ratio)
                @reset st_lux.gen = st_gen

                # Swap population if likelihood of population in new temperature is higher on average
                z_hq[:, :, :, t] .= swap .* z_t1 .+ (1 - swap) .* z_t
                z_hq[:, :, :, t+1] .= (1 - swap) .* z_t1 .+ swap .* z_t
                z_fq .= U.(reshape(z_hq, Q, P, S*num_temps))
            end
        end
    end

    mean_η = clamp.(mean_η ./ num_acceptances, sampler.η_min, sampler.η_max)
    mean_η = ifelse.(isnan.(mean_η), sampler.η, mean_η) |> pu
    
    acceptance_rate = num_acceptances ./ sampler.N
    η_adjustment = ifelse.(acceptance_rate .> target_rate, 
                          sampler.Δη, 
                          one(U) ./ sampler.Δη)
    mean_η = clamp.(mean_η .* η_adjustment, sampler.η_min, sampler.η_max)
    @reset sampler.η = mean_η

    return z_hq, st_lux
end

end
