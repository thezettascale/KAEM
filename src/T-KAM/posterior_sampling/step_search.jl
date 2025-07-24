module autoMALA_StepSearch

export autoMALA_step

using CUDA, KernelAbstractions, Accessors, Lux, LuxCUDA, Statistics, ComponentArrays

using ..Utils
using ..T_KAM_model
using ..HamiltonianMonteCarlo

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
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple;
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

    ẑ, logpos_ẑ, ∇ẑ, p̂, log_r, st_kan, st_lux =
        leapfrog(z, ∇z, x, temps, logpos_z, momentum, M, η_init, model, ps, st_kan, st_lux)

    δ = (log_r .>= log_b) - (log_r .<= log_a)
    active_chains = findall(δ .!= 0) |> cpu_device()
    isempty(active_chains) && return ẑ, logpos_ẑ, ∇ẑ, p̂, η_init, log_r, st

    geq_bool = log_r .>= log_b

    while !isempty(active_chains)

        η_init[active_chains] .=
            safe_step_size_update(η_init[active_chains], δ[active_chains], Δη)

        x_active = seq ? x[:, :, active_chains] : x[:, :, :, active_chains]

        ẑ_active, logpos_ẑ_active, ∇ẑ_active, p̂_active, log_r_active, st_kan, st_lux =
            leapfrog(
                z[:, :, active_chains],
                ∇z[:, :, active_chains],
                x_active,
                temps[active_chains],
                logpos_z[active_chains],
                momentum[:, :, active_chains],
                M[:, :, active_chains],
                η_init[active_chains],
                model,
                ps,
                st_kan,
                st_lux,
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
    return ẑ, logpos_ẑ, ∇ẑ, p̂, η_init, log_r, st_kan, st_lux
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
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
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

    ẑ, logpos_ẑ, ∇ẑ, p̂, η, log_r, st_kan, st_lux = select_step_size(
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
        model,
        ps,
        st_kan,
        st_lux;
        η_min = η_min,
        η_max = η_max,
        seq = seq,
    )

    z_rev, _, _, _, η_prime, _, st_kan, st_lux = select_step_size(
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
        model,
        ps,
        st_kan,
        st_lux;
        η_min = η_min,
        η_max = η_max,
        seq = seq,
    )

    reversible = check_reversibility(z, z_rev, η, η_prime; tol = ε)
    return ẑ, η, η_prime, reversible, log_r, st_kan, st_lux
end

end
