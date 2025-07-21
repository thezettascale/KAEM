module autoMALA_sampling

export initialize_autoMALA_sampler, autoMALA_sample

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
    ∇z::AbstractArray{U},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    logpos_z::AbstractArray{U},
    momentum::AbstractArray{U},
    M::AbstractArray{U},
    η_init::AbstractArray{U},
    Δη::U,
    logpos_withgrad::Function,
    model,
    ps::ComponentArray{T},
    st::NamedTuple;
    η_min::U = full_quant(1e-5),
    η_max::U = one(full_quant),
    seq::Bool = false,
)::Tuple{
    AbstractArray{U},
    AbstractArray{U},
    AbstractArray{U},
    AbstractArray{U},
    AbstractArray{U},
    AbstractArray{U},
    NamedTuple,
} where {T<:half_quant,U<:full_quant}

    ẑ, logpos_ẑ, ∇ẑ, p̂, log_r, st = leapfrop_proposal(
        z,
        ∇z,
        x,
        temps,
        logpos_z,
        momentum,
        M,
        η_init,
        logpos_withgrad,
        model,
        ps,
        st,
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
                ∇z[:, :, active_chains],
                x_active,
                temps[active_chains],
                logpos_z[active_chains],
                momentum[:, :, active_chains],
                M[:, :, active_chains],
                η_init[active_chains],
                logpos_withgrad,
                model,
                ps,
                st,
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
    ∇z::AbstractArray{U},
    x::AbstractArray{T},
    temps::AbstractArray{T},
    logpos_z::AbstractArray{U},
    momentum::AbstractArray{U},
    M::AbstractArray{U},
    logpos_withgrad::Function,
    model,
    ps::ComponentArray{T},
    st::NamedTuple,
    η_init::AbstractArray{U},
    Δη::U,
    η_min::U,
    η_max::U,
    ε::U,
    seq::Bool,
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
        ∇z,
        x,
        temps,
        logpos_z,
        momentum,
        M,
        η_init,
        Δη,
        logpos_withgrad,
        model,
        ps,
        st;
        η_min = η_min,
        η_max = η_max,
        seq = seq,
    )

    z_rev, _, _, _, η_prime, _, st = select_step_size(
        log_a,
        log_b,
        ẑ,
        ∇ẑ,
        x,
        temps,
        logpos_ẑ,
        p̂,
        M,
        η_init,
        Δη,
        logpos_withgrad,
        model,
        ps,
        st;
        η_min = η_min,
        η_max = η_max,
        seq = seq,
    )

    reversible = check_reversibility(z, z_rev, η, η_prime; tol = ε)
    return ẑ, η, η_prime, reversible, log_r, st
end

struct autoMALA_sampler{U<:full_quant}
    compiled_llhood::Any
    compiled_leapfrog::Any
    logpos_withgrad::Any
    compiled_autoMALA_step::Any
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
    model,
    x::AbstractArray{T};
    temps::AbstractArray{T} = [one(half_quant)],
    N::Int = 20,
    N_unadjusted::Int = 1,
    RE_frequency::Int = 10,
    Δη::U = full_quant(2),
    η_min::U = full_quant(1e-5),
    η_max::U = one(full_quant),
    seq::Bool = false,
    compile_mlir::Bool = false,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:half_quant,U<:full_quant}
    z_hq, st_ebm = model.prior.sample_z(model, size(x)[end]*length(temps), ps, st, rng)
    loss_scaling = model.loss_scaling |> U

    num_temps, Q, P, S = length(temps), size(z_hq)[1:2]..., size(x)[end]
    z_hq = reshape(z_hq, Q, P, S, num_temps)
    z_fq = U.(z_hq)
    ∇z_fq = Enzyme.make_zero(z_fq)

    t_expanded = repeat(reshape(temps, 1, num_temps), S, 1) |> device
    seq = model.lkhood.seq_length > 1
    x_t = seq ? repeat(x, 1, 1, 1, num_temps) : repeat(x, 1, 1, 1, 1, num_temps)

    M = zeros(U, Q, P, 1, num_temps) |> device
    ratio_bounds = log.(U.(rand(rng, Uniform(0, 1), S, num_temps, 2))) |> device
    momentum = Enzyme.make_zero(z_fq)
    η = device(st.η_init)

    log_a, log_b = dropdims(minimum(ratio_bounds; dims = 3); dims = 3),
    dropdims(maximum(ratio_bounds; dims = 3); dims = 3)

    compiled_4D_value_and_grad = autoMALA_value_and_grad_4D
    compiled_value_and_grad = autoMALA_value_and_grad
    x_single = seq ? x_t[:, :, :, 1] : x_t[:, :, :, :, 1]

    if compile_mlir
        compiled_4D_value_and_grad = Reactant.@compile autoMALA_value_and_grad_4D(
            z_hq,
            Enzyme.make_zero(z_hq),
            x_t,
            t_expanded,
            model,
            ps,
            st,
        )
        compiled_value_and_grad = Reactant.@compile autoMALA_value_and_grad(
            z_hq[:, :, :, 1],
            Enzyme.make_zero(z_hq[:, :, :, 1]),
            x_single,
            t_expanded[:, 1],
            model,
            ps,
            st,
        )
    end

    function logpos_withgrad(
        z_i::AbstractArray{T},
        x_i::AbstractArray{T},
        t_k::AbstractArray{T},
        m,
        p::ComponentArray{T},
        st_i::NamedTuple,
    )::Tuple{AbstractArray{U},AbstractArray{U},NamedTuple}
        fcn = ndims(z_i) == 4 ? compiled_4D_value_and_grad : compiled_value_and_grad
        logpos, ∇z_k, st_ebm, st_gen = fcn(z_i, Enzyme.make_zero(z_i), x_i, t_k, m, p, st_i, num_temps)
        @reset st_i.ebm = st_ebm
        @reset st_i.gen = st_gen

        return U.(logpos) ./ loss_scaling, U.(∇z_k) ./ loss_scaling, st_i
    end

    compiled_llhood = log_likelihood_MALA
    logpos_z, ∇z_fq, st = logpos_withgrad(z_hq, x_t, t_expanded, model, ps, st)
    compiled_leapfrog = leapfrop_proposal
    compiled_autoMALA_step = autoMALA_step

    if compile_mlir
        compiled_llhood = Reactant.@compile log_likelihood_MALA(
            z_hq[:, :, :, 1],
            x_single,
            model.lkhood,
            ps.gen,
            st.gen,
        )

        compiled_leapfrog = Reactant.@compile leapfrop_proposal(
            z_fq,
            ∇z_fq,
            x_t,
            t_expanded,
            logpos_z,
            momentum,
            device(repeat(M, 1, 1, S, 1)),
            η,
            logpos_withgrad,
            model,
            ps,
            st,
        )

        compiled_autoMALA_step = Reactant.@compile autoMALA_step(
            log_a,
            log_b,
            z_fq,
            ∇z_fq,
            x_t,
            t_expanded,
            logpos_z,
            momentum,
            device(repeat(M, 1, 1, S, 1)),
            logpos_withgrad,
            model,
            ps,
            st,
            η,
            Δη,
            η_min,
            η_max,
            model.ε,
            seq,
        )
    end

    return autoMALA_sampler(
        compiled_llhood,
        compiled_leapfrog,
        logpos_withgrad,
        compiled_autoMALA_step,
        N,
        N_unadjusted,
        Δη,
        η_min,
        η_max,
        RE_frequency,
        seq,
    )
end

function autoMALA_sample(
    sampler,
    model,
    ps::ComponentArray{T},
    st::NamedTuple,
    x::AbstractArray{T};
    temps::AbstractArray{T} = [one(half_quant)],
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
    z_hq, st_ebm = model.prior.sample_z(model, size(x)[end]*length(temps), ps, st, rng)
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
    @reset st.η_init = device(st.η_init)

    log_u = log.(rand(rng, num_temps, sampler.N)) |> device
    ratio_bounds =
        log.(full_quant.(rand(rng, Uniform(0, 1), S, num_temps, 2, sampler.N))) |> device
    log_u_swap = log.(rand(rng, full_quant, S, num_temps, sampler.N)) |> device

    num_acceptances = zeros(Int, S, num_temps) |> device
    mean_η = zeros(full_quant, S, num_temps) |> device
    momentum = Enzyme.make_zero(z_fq)

    burn_in = 0
    η = st.η_init

    for i = 1:sampler.N
        z_cpu = cpu_device()(z_fq)
        for k = 1:num_temps
            momentum[:, :, :, k], M[:, :, 1, k] =
                sample_momentum(z_cpu[:, :, :, k], M[:, :, 1, k])
        end

        log_a, log_b = dropdims(minimum(ratio_bounds[:, :, :, i]; dims = 3); dims = 3),
        dropdims(maximum(ratio_bounds[:, :, :, i]; dims = 3); dims = 3)
        logpos_z, ∇z_fq, st = sampler.logpos_withgrad(z_hq, x_t, t_expanded, model, ps, st)

        if burn_in < sampler.N
            burn_in += 1
            z_fq, logpos_ẑ, ∇ẑ, p̂, log_r, st = sampler.compiled_leapfrog(
                z_fq,
                ∇z_fq,
                x_t,
                t_expanded,
                logpos_z,
                device(momentum),
                device(repeat(M, 1, 1, S, 1)),
                η,
                sampler.logpos_withgrad,
                model,
                ps,
                st,
            )
            z_hq = T.(z_fq)

        else
            ẑ, η_prop, η_prime, reversible, log_r, st = sampler.compiled_autoMALA_step(
                log_a,
                log_b,
                z_fq,
                ∇z_fq,
                x_t,
                t_expanded,
                logpos_z,
                device(momentum),
                device(repeat(M, 1, 1, S, 1)),
                sampler.logpos_withgrad,
                model,
                ps,
                st,
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
                    z_t = z_hq[:, :, :, t]
                    z_t1 = z_hq[:, :, :, t+1]
                    ll_t, st_gen =
                        sampler.compiled_llhood(z_t, x, model.lkhood, ps.gen, st.gen)
                    ll_t1, st_gen =
                        sampler.compiled_llhood(z_t1, x, model.lkhood, ps.gen, st_gen)
                    log_swap_ratio = (temps[t+1] - temps[t]) .* (ll_t - ll_t1)

                    swap = log_u_swap[t, i] < mean(log_swap_ratio)
                    @reset st.gen = st_gen

                    # Swap population if likelihood of population in new temperature is higher on average
                    if swap
                        z_copy = z_t
                        z_hq[:, :, :, t] .= z_t1
                        z_hq[:, :, :, t+1] .= z_copy
                        z_fq = full_quant.(z_hq)
                    end
                end
            end
        end


    end

    mean_η = clamp.(mean_η ./ num_acceptances, sampler.η_min, sampler.η_max)
    mean_η = ifelse.(isnan.(mean_η), st.η_init, mean_η) |> device
    @reset st.η_init = mean_η

    return z_hq, st
end

end
