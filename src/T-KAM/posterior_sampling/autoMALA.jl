module autoMALA_sampling

export autoMALA_sampler

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
include("preconditioner.jl")
include("../gen/gen_model.jl")
include("hamiltonian.jl")
include("log_posteriors.jl")
using .Utils: device, half_quant, full_quant
using .Preconditioning
using .HamiltonianDynamics
using .LogPosteriors: autoMALA_logpos_4D, autoMALA_logpos
using .GeneratorModel: log_likelihood_MALA

function safe_step_size_update(
    η::AbstractArray{U},
    δ::AbstractArray{Int},
    Δη::U,
) where {U<:full_quant}
    η_new = η .* Δη .^ δ
    return ifelse.(isfinite.(η_new), η_new, η)
end

function check_reversibility(
    ẑ::AbstractArray{U},
    z::AbstractArray{U},
    η::AbstractArray{U},
    η_prime::AbstractArray{U};
    tol::U = full_quant(1e-6),
) where {U<:full_quant}
    # Both checks may be required to maintain detailed balance
    # pos_diff = dropdims(maximum(abs.(ẑ - z); dims=(1,2)); dims=(1,2)) .< tol * maximum(abs.(z)) # leapfrog reversibility check
    step_diff = abs.(η - η_prime) .< tol .* η # autoMALA reversibility check
    return step_diff
end

function select_step_size(
    log_a::AbstractArray{U},
    log_b::AbstractArray{U},
    z::AbstractArray{U},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    st::NamedTuple,
    logpos_z::AbstractArray{U},
    ∇z::AbstractArray{U},
    momentum::AbstractArray{U},
    M::AbstractArray{U},
    η_init::AbstractArray{U},
    Δη::U,
    logpos_withgrad::Function;
    η_min::U = full_quant(1e-5),
    η_max::U = one(full_quant),
    seq::Bool = false,
)::Tuple{
    AbstractArray{U},
    AbstractArray{U},
    AbstractArray{U},
    AbstractArray{U},
    AbstractArray{U},
    NamedTuple,
} where {T<:half_quant,U<:full_quant}

    ẑ, logpos_ẑ, ∇ẑ, p̂, log_r, st = leapfrop_proposal(
        z,
        x,
        st,
        logpos_z,
        ∇z,
        momentum,
        M,
        η_init,
        logpos_withgrad,
        temps,
    )

    δ = (log_r .>= log_b) - (log_r .<= log_a)
    active_chains = findall(δ .!= 0) |> cpu_device()
    isempty(active_chains) && return ẑ, logpos_ẑ, ∇ẑ, p̂, η_init, log_r, st

    geq_bool = log_r .>= log_b

    while !isempty(active_chains)

        η_init[active_chains] .=
            safe_step_size_update(η_init[active_chains], δ[active_chains], Δη)

        x_active = seq ? x[:, :, active_chains] : x[:, :, :, active_chains]
        ẑ_active, logpos_ẑ_active, ∇ẑ_active, p̂_active, log_r_active, st =
            leapfrop_proposal(
                z[:, :, active_chains],
                x_active,
                st,
                logpos_z[active_chains],
                ∇z[:, :, active_chains],
                momentum[:, :, active_chains],
                M[:, :, active_chains],
                η_init[active_chains],
                logpos_withgrad,
                temps[active_chains],
            )

        ẑ[:, :, active_chains] .= ẑ_active
        logpos_ẑ[active_chains] .= logpos_ẑ_active
        ∇ẑ[:, :, active_chains] .= ∇ẑ_active
        p̂[:, :, active_chains] .= p̂_active
        log_r[active_chains] .= log_r_active

        δ[active_chains] .= ifelse.(
            δ[active_chains] .== 1 .&& log_r[active_chains] .< log_b[active_chains],
            0,
            δ[active_chains],
        )
        δ[active_chains] .= ifelse.(
            δ[active_chains] .== -1 .&& log_r[active_chains] .> log_a[active_chains],
            0,
            δ[active_chains],
        )
        δ[active_chains] .= ifelse.(isnan.(log_r[active_chains]), 0, δ[active_chains])
        δ[active_chains] .=
            ifelse.(η_min .< η_init[active_chains] .< η_max, δ[active_chains], 0)
        active_chains = findall(δ .!= 0) |> cpu_device()
    end

    # Reduce step size for chains that initially had too high acceptance with safety check
    η_init = safe_step_size_update(η_init, -1 .* geq_bool, Δη)
    return ẑ, logpos_ẑ, ∇ẑ, p̂, η_init, log_r, st
end

function autoMALA_step(
    log_a::AbstractArray{U},
    log_b::AbstractArray{U},
    z::AbstractArray{U},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    st::NamedTuple,
    logpos_z::AbstractArray{U},
    ∇z::AbstractArray{U},
    momentum::AbstractArray{U},
    M::AbstractArray{U},
    η_init::AbstractArray{U},
    Δη::U,
    logpos_withgrad::Function;
    η_min::U = full_quant(1e-5),
    η_max::U = one(full_quant),
    ε::U = eps(full_quant),
    seq::Bool = false,
)::Tuple{
    AbstractArray{U},
    AbstractArray{U},
    AbstractArray{U},
    AbstractArray{U},
    AbstractArray{U},
    NamedTuple,
} where {T<:half_quant,U<:full_quant}

    ẑ, logpos_ẑ, ∇ẑ, p̂, η, log_r, _ = select_step_size(
        log_a,
        log_b,
        z,
        x,
        temps,
        st,
        logpos_z,
        ∇z,
        momentum,
        M,
        η_init,
        Δη,
        logpos_withgrad;
        η_min = η_min,
        η_max = η_max,
        seq = seq,
    )

    z_rev, _, _, _, η_prime, _, st = select_step_size(
        log_a,
        log_b,
        ẑ,
        x,
        temps,
        st,
        logpos_ẑ,
        ∇ẑ,
        p̂,
        M,
        η_init,
        Δη,
        logpos_withgrad;
        η_min = η_min,
        η_max = η_max,
        seq = seq,
    )

    reversible = check_reversibility(z, z_rev, η, η_prime; tol = ε)
    return ẑ, η, η_prime, reversible, log_r, st
end

function autoMALA_sampler(
    model::Any,
    ps::ComponentArray{T},
    st::NamedTuple,
    x::AbstractArray{T};
    temps::AbstractArray{T} = [one(half_quant)],
    N::Int = 20,
    N_unadjusted::Int = 1,
    Δη::U = full_quant(2),
    η_min::U = full_quant(1e-5),
    η_max::U = one(full_quant),
    RE_frequency::Int = 10,
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant,U<:full_quant}
    """
    Metropolis-adjusted Langevin algorithm (MALA) sampler to generate posterior samples.

    Args:
        m: The model.
        ps: The parameters of the model.
        st: The states of the model.
        x: The data
        t: The temperatures if using Thermodynamic Integration.
        N: The number of iterations.
        η_init: The initial step size.
        rng: The random number generator.

    Returns:
        The posterior samples.
    """

    # Initialize from prior (already in bounded space)
    z, st_ebm = model.prior.sample_z(model, size(x)[end]*length(temps), ps, st, rng)
    z = U.(z)
    loss_scaling = model.loss_scaling |> U

    num_temps, Q, P, S = length(temps), size(z)[1:2]..., size(x)[end]
    z = reshape(z, Q, P, S, num_temps)

    t_expanded = repeat(reshape(temps, 1, num_temps), S, 1) |> device
    seq = model.lkhood.seq_length > 1
    x_t = seq ? repeat(x, 1, 1, 1, num_temps) : repeat(x, 1, 1, 1, 1, num_temps)

    # Initialize preconditioner
    M = zeros(U, Q, P, 1, num_temps)
    z_cpu = cpu_device()(z)
    for k = 1:num_temps
        M[:, :, 1, k] = init_mass_matrix(view(z_cpu,:,:,:,k))
    end
    @reset st.η_init = device(st.η_init)

    # Pre-allocate noise
    log_u = log.(rand(rng, U, S, num_temps, N)) |> device
    ratio_bounds = log.(U.(rand(rng, Uniform(0, 1), S, num_temps, 2, N))) |> device
    log_u_swap = log.(rand(rng, U, S, num_temps, N)) |> device


    num_acceptances = zeros(Int, S, num_temps) |> device
    mean_η = zeros(U, S, num_temps) |> device
    momentum = similar(z) |> cpu_device()

    function logpos_4D(
        z_i::AbstractArray{T},
        x_i::AbstractArray{T},
        t_k::AbstractArray{T},
        s::NamedTuple,
        m::Any,
        p::ComponentArray{T},
    )::T
        sum(
            first(
                autoMALA_logpos_4D(
                    z_i,
                    x_i,
                    t_k,
                    s,
                    m,
                    p;
                    rng = rng,
                    num_temps = num_temps,
                ),
            ),
        )
    end
    function logpos_2D(
        z_i::AbstractArray{T},
        x_i::AbstractArray{T},
        t_k::AbstractArray{T},
        s::NamedTuple,
        m::Any,
        p::ComponentArray{T},
    )::T
        sum(
            first(
                autoMALA_logpos(z_i, x_i, t_k, s, m, p; rng = rng, num_temps = num_temps),
            ),
        )
    end

    function logpos_withgrad(
        z_i::AbstractArray{T},
        x_i::AbstractArray{T},
        st_i::NamedTuple,
        t_k::AbstractArray{T},
    )::Tuple{U,AbstractArray{U},NamedTuple}
        fcn = ndims(z_i) == 4 ? logpos_4D : logpos_2D
        ∇z = zeros(T, size(z_i)) |> device
        CUDA.@fastmath Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            fcn,
            Enzyme.Active,
            Enzyme.Duplicated(T.(z_i), ∇z),
            Enzyme.Const(x_i),
            Enzyme.Const(t_k),
            Enzyme.Const(Lux.testmode(st_i)),
            Enzyme.Const(model),
            Enzyme.Const(ps),
        )
        logpos_z, st_ebm, st_gen =
            CUDA.@fastmath fcn(T.(z_i), x_i, t_k, st_i, model, ps)
        @reset st_i.ebm = st_ebm
        @reset st_i.gen = st_gen
        return U.(logpos_z) ./ loss_scaling, U.(∇z) ./ loss_scaling, st_i
    end

    burn_in = 0
    η = st.η_init

    for i = 1:N
        z_cpu = cpu_device()(z)
        for k = 1:num_temps
            momentum[:, :, :, k], M[:, :, 1, k] =
                sample_momentum(z_cpu[:, :, :, k], M[:, :, 1, k])
        end

        log_a, log_b = dropdims(minimum(ratio_bounds[:, :, :, i]; dims = 3); dims = 3),
        dropdims(maximum(ratio_bounds[:, :, :, i]; dims = 3); dims = 3)
        logpos_z, ∇z, st = logpos_withgrad(z, x_t, st, t_expanded)

        if burn_in < N_unadjusted
            burn_in += 1
            z, logpos_ẑ, ∇ẑ, p̂, log_r, st = leapfrop_proposal(
                z,
                x_t,
                st,
                logpos_z,
                device(∇z),
                device(momentum),
                device(repeat(M, 1, 1, S, 1)),
                η,
                logpos_withgrad,
                t_expanded,
            )
        else
            ẑ, η_prop, η_prime, reversible, log_r, st = autoMALA_step(
                log_a,
                log_b,
                z,
                x_t,
                t_expanded,
                st,
                logpos_z,
                ∇z,
                device(momentum),
                device(repeat(M, 1, 1, S, 1)),
                η,
                U(Δη),
                logpos_withgrad;
                η_min = η_min,
                η_max = η_max,
                ε = U(model.ε),
                seq = seq,
            )

            accept = (log_u[:, :, i] .< log_r) .* reversible
            z =
                ẑ .* reshape(accept, 1, 1, S, num_temps) +
                z .* reshape(1 .- accept, 1, 1, S, num_temps)
            mean_η .= mean_η .+ η_prop .* accept
            η .= η_prop .* accept .+ η .* (1 .- accept)
            num_acceptances .= num_acceptances .+ accept

            # Replica exchange Monte Carlo
            if i % RE_frequency == 0 && num_temps > 1
                for t = 1:(num_temps-1)

                    # Global swap criterion
                    z_hq = T.(z)
                    ll_t, st_gen = log_likelihood_MALA(
                        z_hq[:, :, :, t],
                        x,
                        model.lkhood,
                        ps.gen,
                        st.gen;
                        rng = rng,
                        ε = model.ε,
                    )
                    ll_t1, st_gen = log_likelihood_MALA(
                        z_hq[:, :, :, t+1],
                        x,
                        model.lkhood,
                        ps.gen,
                        st_gen;
                        rng = rng,
                        ε = model.ε,
                    )
                    log_swap_ratio = (temps[t+1] - temps[t]) .* (ll_t - ll_t1)

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


    end

    mean_η = clamp.(mean_η ./ num_acceptances, η_min, η_max)
    mean_η = ifelse.(isnan.(mean_η), st.η_init, mean_η) |> device
    @reset st.η_init = mean_η

    any(isnan.(z)) && error("NaN in z")
    return T.(z), st
end

end
