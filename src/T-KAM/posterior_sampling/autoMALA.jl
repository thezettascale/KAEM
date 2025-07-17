module autoMALA_sampling

export initialize_autoMALA_sampler, sample

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
include("preconditioner.jl")
include("../gen/gen_model.jl")
include("hamiltonian.jl")
include("log_posteriors.jl")
using .Utils: device, half_quant, full_quant
using .Preconditioning
using .HamiltonianDynamics
using .LogPosteriors: autoMALA_value_and_grad_4D, autoMALA_value_and_grad
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

struct autoMALA_sampler{T}
    compiled_llhood::Function
    compiled_logpos_withgrad::Function
    compiled_autoMALA_step::Function
    N::Int
    N_unadjusted::Int
    Δη::U
    η_min::U
    η_max::U
    RE_frequency::Int
    seq::Bool
end

function initialize_autoMALA_sampler(
    ps::ComponentArray{T},
    st::NamedTuple,
    model::Any,
    x::AbstractArray{T};
    temps::AbstractArray{T} = [one(half_quant)],
    N::Int = 20,
    N_unadjusted::Int = 1,
    Δη::U = full_quant(2),
    η_min::U = full_quant(1e-5),
    η_max::U = one(full_quant),
    seq::Bool = false,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:half_quant,U<:full_quant}
    z, st_ebm = model.prior.sample_z(model, size(x)[end]*length(temps), ps, st, rng)
    z = U.(z)
    loss_scaling = model.loss_scaling |> U

    num_temps, Q, P, S = length(temps), size(z)[1:2]..., size(x)[end]
    z = reshape(z, Q, P, S, num_temps)

    t_expanded = repeat(reshape(temps, 1, num_temps), S, 1) |> device
    seq = model.lkhood.seq_length > 1
    x_t = seq ? repeat(x, 1, 1, 1, num_temps) : repeat(x, 1, 1, 1, 1, num_temps)

    M = zeros(U, Q, P, 1, num_temps) |> device
    ratio_bounds = log.(U.(rand(rng, Uniform(0, 1), S, num_temps, 2))) |> device
    momentum = similar(z) |> device
    η = device(st.η_init)

    compiled_logpos_withgrad_4D = Reactant.@compile autoMALA_value_and_grad_4D(
        z,
        ∇z,
        x_t,
        t_expanded,
        st,
        model,
        ps,
        num_temps,
        seq,
    )
    compiled_logpos_withgrad_3D = Reactant.@compile autoMALA_value_and_grad(
        z[:, :, :, 1],
        ∇z[:, :, :, 1],
        x_t[:, :, :, 1],
        t_expanded[:, :, :, 1],
        st,
        model,
        ps,
        num_temps,
        seq,
    )
    compiled_llhood = Reactant.@compile log_likelihood_MALA(
        z[:, :, :, 1],
        x,
        model.lkhood,
        ps.gen,
        st.gen,
    )

    function logpos_withgrad(
        z_i::AbstractArray{T},
        x_i::AbstractArray{T},
        st_i::NamedTuple,
        t_k::AbstractArray{T},
        m::Any,
        p::ComponentArray{T},
    )::Tuple{U,AbstractArray{U},NamedTuple}
        fcn = ndims(z_i) == 4 ? compiled_logpos_withgrad_4D : compiled_logpos_withgrad_3D
        logpos, ∇z, st_ebm, st_gen = fcn(z_i, x_i, t_k, st_i, m, p, num_temps, seq)
        @reset st_i.ebm = st_ebm
        @reset st_i.gen = st_gen
        return U.(logpos) ./ loss_scaling, U.(∇z) ./ loss_scaling, st_i
    end

    logpos_z, ∇z, st = logpos_withgrad(z, x_t, st, t_expanded, model, ps)
    compiled_autoMALA_step = Reactant.@compile autoMALA_step(
        log_a,
        log_b,
        z,
        x_t,
        t_expanded,
        st,
        logpos_z,
        ∇z,
        momentum,
        M,
        η,
        Δη,
        logpos_withgrad,
    )

    log_a, log_b = dropdims(minimum(ratio_bounds; dims = 3); dims = 3),
    dropdims(maximum(ratio_bounds; dims = 3); dims = 3)

    return autoMALA_sampler(
        compiled_llhood,
        compiled_logpos_withgrad,
        compiled_autoMALA_step,
        N,
        N_unadjusted,
        Δη,
        η_min,
        η_max,
        RE_frequency,
        ε,
        seq,
    )
end

function sample(
    sampler::autoMALA_sampler,
    model::Any,
    ps::ComponentArray{T},
    st::NamedTuple,
    x::AbstractArray{T};
    temps::AbstractArray{T} = [one(half_quant)],
    rng::AbstractRNG = Random.default_rng(),
) where {T<:half_quant,U<:full_quant}
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
    z, st_ebm = model.prior.sample_z(model, size(x)[end]*length(temps), ps, st, rng)
    z = U.(z)
    loss_scaling = model.loss_scaling |> U

    num_temps, Q, P, S = length(temps), size(z)[1:2]..., size(x)[end]
    z = reshape(z, Q, P, S, num_temps)
    ∇z = similar(z) |> device

    t_expanded = repeat(reshape(temps, 1, num_temps), S, 1) |> device
    x_t = sampler.seq ? repeat(x, 1, 1, 1, num_temps) : repeat(x, 1, 1, 1, 1, num_temps)

    # Initialize preconditioner
    M = zeros(U, Q, P, 1, num_temps)
    z_cpu = cpu_device()(z)
    for k = 1:num_temps
        M[:, :, 1, k] = init_mass_matrix(view(z_cpu,:,:,:,k))
    end
    @reset st.η_init = device(st.η_init)

    log_u = log.(rand(rng, U, S, num_temps, sampler.N)) |> device
    ratio_bounds = log.(U.(rand(rng, Uniform(0, 1), S, num_temps, 2, sampler.N))) |> device
    log_u_swap = log.(rand(rng, U, S, num_temps, sampler.N)) |> device

    num_acceptances = zeros(Int, S, num_temps) |> device
    mean_η = zeros(U, S, num_temps) |> device
    momentum = similar(z) |> device

    burn_in = 0
    η = st.η_init

    for i = 1:sampler.N
        z_cpu = cpu_device()(z)
        for k = 1:num_temps
            momentum[:, :, :, k], M[:, :, 1, k] =
                sample_momentum(z_cpu[:, :, :, k], M[:, :, 1, k])
        end

        log_a, log_b = dropdims(minimum(ratio_bounds[:, :, :, i]; dims = 3); dims = 3),
        dropdims(maximum(ratio_bounds[:, :, :, i]; dims = 3); dims = 3)
        logpos_z, ∇z, st = logpos_withgrad(z, x_t, st, t_expanded, model, ps)

        if sampler.N_unadjusted < sampler.N
            z, logpos_ẑ, ∇ẑ, p̂, log_r, st = leapfrop_proposal(
                z,
                x_t,
                st,
                logpos_z,
                ∇z,
                device(momentum),
                device(repeat(M, 1, 1, S, 1)),
                η,
                sampler.Δη,
                sampler.η_min,
                sampler.η_max,
                sampler.ε,
            )
        else
            ẑ, η_prop, η_prime, reversible, log_r, st = sampler.compiled_autoMALA_step(
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
                sampler.Δη,
                sampler.η_min,
                sampler.η_max,
                sampler.ε,
            )
        end

        accept = (log_u[:, :, i] .< log_r) .* reversible
        z =
            ẑ .* reshape(accept, 1, 1, S, num_temps) .+
            z .* reshape(1 .- accept, 1, 1, S, num_temps)
        mean_η .= mean_η .+ η_prop .* accept
        η .= η_prop .* accept .+ η .* (1 .- accept)
        num_acceptances .= num_acceptances .+ accept

        # Replica exchange Monte Carlo
        if i % sampler.RE_frequency == 0 && num_temps > 1
            for t = 1:(num_temps-1)

                # Global swap criterion
                z_hq = T.(z)
                ll_t, st_gen = sampler.compiled_llhood(
                    z_hq[:, :, :, t],
                    x,
                    model.lkhood,
                    ps.gen,
                    st.gen,
                )
                ll_t1, st_gen = sampler.compiled_llhood(
                    z_hq[:, :, :, t+1],
                    x,
                    model.lkhood,
                    ps.gen,
                    st_gen,
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

    mean_η = clamp.(mean_η ./ num_acceptances, sampler.η_min, sampler.η_max)
    mean_η = ifelse.(isnan.(mean_η), st.η_init, mean_η) |> device
    @reset st.η_init = mean_η

    any(isnan.(z)) && error("NaN in z")
    return T.(z), st
end

end
