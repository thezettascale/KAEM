module autoMALA_sampling

export initialize_autoMALA_sampler, autoMALA_sampler

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
    ComponentArrays

include("../../utils.jl")
include("../gen/gen_model.jl")
include("preconditioner.jl")
include("step_search.jl")
include("hmc_updates.jl")
include("log_posteriors.jl")
using .Utils: device, half_quant, full_quant
using .Preconditioning
using .HamiltonianMonteCarlo: leapfrog
using .autoMALA_StepSearch: autoMALA_step
using .LogPosteriors: logpos_withgrad
using .GeneratorModel: log_likelihood_MALA

struct autoMALA_sampler{U<:full_quant}
    N::Int
    N_unadjusted::Int
    η::AbstractArray{U}
    Δη::U
    η_min::U
    η_max::U
    RE_frequency::Int
    seq::Bool
end

function initialize_autoMALA_sampler(;
    N::Int = 20,
    N_unadjusted::Int = 1,
    RE_frequency::Int = 10,
    η::U = full_quant(1e-3),
    Δη::U = full_quant(2),
    η_min::U = full_quant(1e-5),
    η_max::U = one(full_quant),
    seq::Bool = false,
    samples::Int = 100,
) where {T<:half_quant,U<:full_quant}

    return autoMALA_sampler(
        N,
        N_unadjusted,
        repeat([η], samples, num_temps) |> device,
        Δη,
        η_min,
        η_max,
        RE_frequency,
        seq,
    )
end

function (sampler::autoMALA_sampler)(
    model,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    x::AbstractArray{T};
    temps::AbstractArray{T} = [one(T)],
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
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
    z_hq, st_ebm =
        model.prior.sample_z(model, size(x)[end]*length(temps), ps, st_kan, st_lux, rng)
    loss_scaling = model.loss_scaling |> full_quant

    num_temps, Q, P, S = length(temps), size(z_hq)[1:2]..., size(x)[end]
    z_hq = reshape(z_hq, Q, P, S, num_temps)

    # Pre-allocate for both precisions
    z_fq = full_quant.(z_hq)
    ∇z_fq = Enzyme.make_zero(z_fq)
    z_copy = similar(z_hq[:, :, :, 1]) |> device
    z_t, z_t1 = z_copy, z_copy

    t_expanded = repeat(reshape(temps, 1, num_temps), S, 1) |> device
    x_t = sampler.seq ? repeat(x, 1, 1, 1, num_temps) : repeat(x, 1, 1, 1, 1, num_temps)

    # Initialize preconditioner
    M = zeros(full_quant, Q, P, 1, num_temps)
    z_cpu = cpu_device()(z_fq)
    for k = 1:num_temps
        M[:, :, 1, k] = init_mass_matrix(view(z_cpu,:,:,:,k))
    end
    @reset model.η = device(model.η)

    log_u = log.(rand(rng, num_temps, sampler.N)) |> device
    ratio_bounds =
        log.(full_quant.(rand(rng, Uniform(0, 1), S, num_temps, 2, sampler.N))) |> device
    log_u_swap = log.(rand(rng, full_quant, S, num_temps-1, sampler.N)) |> device

    num_acceptances = zeros(Int, S, num_temps) |> device
    mean_η = zeros(full_quant, S, num_temps) |> device
    momentum = Enzyme.make_zero(z_fq)

    burn_in = 0
    η = sampler.η

    for i = 1:sampler.N
        z_cpu = cpu_device()(z_fq)
        for k = 1:num_temps
            momentum[:, :, :, k], M[:, :, 1, k] =
                sample_momentum(z_cpu[:, :, :, k], M[:, :, 1, k])
        end

        log_a, log_b = dropdims(minimum(ratio_bounds[:, :, :, i]; dims = 3); dims = 3),
        dropdims(maximum(ratio_bounds[:, :, :, i]; dims = 3); dims = 3)
        logpos_z, ∇z_fq, st_kan, st_lux = logpos_withgrad(z_hq, x_t, t_expanded, model, ps, st_kan, st_lux)

        if burn_in < sampler.N
            burn_in += 1
            z_fq, logpos_ẑ, ∇ẑ, p̂, log_r, st_kan, st_lux = leapfrog(
                z_fq,
                ∇z_fq,
                x_t,
                t_expanded,
                logpos_z,
                device(momentum),
                device(repeat(M, 1, 1, S, 1)),
                η,
                model,
                ps,
                st_kan,
                st_lux,
            )
            z_hq = T.(z_fq)

        else
            ẑ, η_prop, η_prime, reversible, log_r, st_kan, st_lux = autoMALA_step(
                log_a,
                log_b,
                z_fq,
                ∇z_fq,
                x_t,
                t_expanded,
                logpos_z,
                device(momentum),
                device(repeat(M, 1, 1, S, 1)),
                model,
                ps,
                st_kan,
                st_lux,
                η,
                sampler.Δη,
                sampler.η_min,
                sampler.η_max,
                model.ε,
                sampler.seq,
            )

            accept = (log_u[:, :, i] .< log_r) .* reversible
            z_fq =
                ẑ .* reshape(accept, 1, 1, S, num_temps) .+
                z_fq .* reshape(1 .- accept, 1, 1, S, num_temps)
            mean_η .= mean_η .+ η_prop .* accept
            η .= η_prop .* accept .+ η .* (1 .- accept)
            num_acceptances .= num_acceptances .+ accept

            z_hq = T.(z_fq)

            # Replica exchange Monte Carlo
            if i % sampler.RE_frequency == 0 && num_temps > 1
                for t = 1:(num_temps-1)

                    # Global swap criterion
                    z_t = copy(z_hq[:, :, :, t])
                    z_t1 = copy(z_hq[:, :, :, t+1])
                    ll_t, st_gen = log_likelihood_MALA(
                        z_t,
                        x,
                        model.lkhood,
                        ps.gen,
                        st_kan.gen,
                        st_lux.gen;
                        ε = model.ε,
                    )
                    ll_t1, st_gen = log_likelihood_MALA(
                        z_t1,
                        x,
                        model.lkhood,
                        ps.gen,
                        st_kan.gen,
                        st_lux.gen;
                        ε = model.ε,
                    )
                    log_swap_ratio = (temps[t+1] - temps[t]) .* (ll_t - ll_t1)

                    swap = T(log_u_swap[t, i] < mean(log_swap_ratio))
                    @reset st_lux.gen = st_gen

                    # Swap population if likelihood of population in new temperature is higher on average
                    z_hq[:, :, :, t] .= swap .* z_t1 .+ (1 - swap) .* z_t
                    z_hq[:, :, :, t+1] .= (1 - swap) .* z_t1 .+ swap .* z_t
                    z_fq = full_quant.(z_hq)
                end
            end
        end
    end

    mean_η = clamp.(mean_η ./ num_acceptances, sampler.η_min, sampler.η_max)
    mean_η = ifelse.(isnan.(mean_η), sampler.η, mean_η) |> device
    @reset sampler.η = mean_η

    return z_hq, st_lux
end

end
