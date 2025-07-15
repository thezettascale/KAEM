module ULA_sampling

export ULA_sampler

using CUDA,
    KernelAbstractions,
    LinearAlgebra,
    Random,
    Lux,
    LuxCUDA,
    Distributions,
    Accessors,
    Statistics,
    DifferentiationInterface,
    Enzyme,
    ComponentArrays

include("../../utils.jl")
include("../gen/gen_model.jl")
include("log_posteriors.jl")
using .Utils: device, next_rng, half_quant, full_quant, fq
using .GeneratorModel: log_likelihood_MALA
using .LogPosteriors: unadjusted_logpos

π_dist = Dict(
    "uniform" => (p, b, rng) -> rand(rng, p, 1, b),
    "gaussian" => (p, b, rng) -> randn(rng, p, 1, b),
    "lognormal" => (p, b, rng) -> rand(rng, LogNormal(0, 1), p, 1, b),
    "ebm" => (p, b, rng) -> randn(rng, p, 1, b),
)

function ULA_sampler(
    model::Any,
    ps::ComponentArray{T},
    st::NamedTuple,
    x::AbstractArray{T};
    temps::AbstractArray{T} = [one(half_quant)],
    N::Int = 20,
    seed::Int = 1,
    RE_frequency::Int = 10,
    prior_sampling_bool::Bool = false,
    prior_η::U = full_quant(1e-3),
    num_samples::Int = 100,
)::Tuple{AbstractArray{T},NamedTuple,Int} where {T<:half_quant,U<:full_quant}
    """
    Unadjusted Langevin Algorithm (ULA) sampler to generate posterior samples.

    Args:
        m: The model.
        ps: The parameters of the model.
        st: The states of the model.
        x: The data.
        t: The temperatures if using Thermodynamic Integration.
        N: The number of iterations.
        seed: The seed for the random number generator.

        
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
        if model.prior.ula && prior_sampling_bool
            seed, rng = next_rng(seed)
            z =
                π_dist[model.prior.prior_type](model.prior.p_size, num_samples, rng) .|>
                U |>
                device
        else
            z, st_ebm, seed =
                model.prior.sample_z(m, size(x)[end]*length(temps), ps, st, seed)
            @reset st.ebm = st_ebm
            z .|> U
        end
    end

    loss_scaling = model.loss_scaling |> U

    η = prior_sampling_bool ? prior_η : mean(st.η_init)
    seq = model.lkhood.seq_length > 1

    num_temps, Q, P, S = length(temps), size(z)[1:2]..., size(x)[end]
    S = prior_sampling_bool ? size(z)[end] : S
    z = reshape(z, Q, P, S, num_temps)
    ∇z = zeros(T, size(z)) |> device

    logpos_fcn =
        (z_i, x_i, t_i, m, p, s, seed) -> begin
            first(
                unadjusted_logpos(
                    z_i,
                    x_i,
                    t_i,
                    m,
                    p,
                    s,
                    seed;
                    prior_sampling_bool = prior_sampling_bool,
                ),
            )
        end

    # Pre-allocate noise
    seed, rng = next_rng(seed)
    noise = randn(rng, U, Q, P, S, num_temps, N)
    seed, rng = next_rng(seed)
    log_u_swap = log.(rand(rng, U, S, num_temps, N)) |> device

    logpos_grad =
        (z_i) -> begin
            CUDA.@fastmath Enzyme.autodiff(
                set_runtime_activity(Reverse),
                logpos_fcn,
                Enzyme.Active,
                Enzyme.Duplicated(T.(z_i), ∇z),
                Enzyme.Const(x),
                Enzyme.Const(m),
                Enzyme.Const(ps),
                Enzyme.Const(st),
                Enzyme.Const(seed),
            )

            _, st_ebm, st_gen, seed =
                CUDA.@fastmath logpos_fcn(T.(z_i), x, temps, m, ps, st, seed)
            @reset st.ebm = st_ebm
            @reset st.gen = st_gen
            return U.(∇z) ./ loss_scaling
        end

    pos_before =
        CUDA.@fastmath first(log_posterior(T.(z), Lux.testmode(st))) ./ loss_scaling
    for i = 1:N
        ξ = device(noise[:, :, :, :, i])
        z += η .* logpos_grad(z) .+ sqrt(2 * η) .* ξ

        if i % RE_frequency == 0 && num_temps > 1 && !prior_sampling_bool
            z_hq = T.(z)
            for t = 1:(num_temps-1)
                ll_t, st_gen = log_llhood_fcn(z_hq[:, :, :, t], st.gen, temps[end])
                ll_t1, st_gen = log_llhood_fcn(z_hq[:, :, :, t+1], st_gen, temps[end])
                log_swap_ratio = dropdims(
                    sum((temps[t+1] - temps[t]) .* (ll_t - ll_t1); dims = 1);
                    dims = 1,
                )
                swap = log_u_swap[:, t, i] .< log_swap_ratio
                @reset st.gen = st_gen

                # Swap samples where accepted
                z[:, :, :, t] .=
                    z[:, :, :, t] .* reshape(swap, 1, 1, S) +
                    z[:, :, :, t+1] .* reshape(1 .- swap, 1, 1, S)
                z[:, :, :, t+1] .=
                    z[:, :, :, t+1] .* reshape(swap, 1, 1, S) +
                    z[:, :, :, t] .* reshape(1 .- swap, 1, 1, S)
            end
        end
    end

    pos_after = CUDA.@fastmath first(log_posterior(T.(z), Lux.testmode(st))) ./ loss_scaling
    dist = prior_sampling_bool ? "Prior" : "Posterior"
    model.verbose && println("$(dist) change: $(pos_after - pos_before)")

    if prior_sampling_bool
        st = st.ebm
        z = dropdims(z; dims = 4)
    end

    return T.(z), st, seed
end

end
