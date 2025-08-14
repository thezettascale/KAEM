module autoMALA_StepSearch

export autoMALA_step

using CUDA, Accessors, Lux, LuxCUDA, Statistics, ComponentArrays

using ..Utils
using ..T_KAM_model
using ..LangevinUpdates

function safe_step_size_update(
    η::AbstractArray{U,1},
    δ::AbstractArray{Int},
    Δη::U,
) where {U<:full_quant}
    η_new = η .* Δη .^ δ
    return ifelse.(isfinite.(η_new), η_new, η)
end

function check_reversibility(
    ẑ::AbstractArray{U,3},
    z::AbstractArray{U,3},
    η::AbstractArray{U,1},
    η_prime::AbstractArray{U,1};
    tol::U = full_quant(1e-3),
)::AbstractArray{Bool,1} where {U<:full_quant}
    # Check both position differences and step size differences for detailed balance
    # pos_diff = dropdims(maximum(abs.(ẑ - z); dims=(1,2)); dims=(1,2)) .< tol * maximum(abs.(z))
    step_diff = abs.(η - η_prime) .< tol .* η
    return step_diff
end

function select_step_size(
    log_a::AbstractArray{U,1},
    log_b::AbstractArray{U,1},
    z::AbstractArray{U,3},
    ∇z::AbstractArray{U,3},
    x::AbstractArray{T},
    temps::AbstractArray{T,1},
    logpos_z::AbstractArray{U,1},
    momentum::AbstractArray{U,3},
    M::AbstractArray{U,2},
    η_init::AbstractArray{U,1},
    Δη::U,
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple;
    η_min::U = full_quant(1e-5),
    η_max::U = one(full_quant),
)::Tuple{
    AbstractArray{U,3},
    AbstractArray{U,1},
    AbstractArray{U,3},
    AbstractArray{U,3},
    AbstractArray{U,1},
    AbstractArray{U,1},
    NamedTuple,
} where {T<:half_quant,U<:full_quant}

    ẑ, logpos_ẑ, ∇ẑ, p̂, log_r, st_lux =
        leapfrog(z, ∇z, x, temps, logpos_z, momentum, M, η_init, model, ps, st_kan, st_lux)

    δ = (log_r .>= log_b) - (log_r .<= log_a)
    active_chains = findall(δ .!= 0) |> cpu_device()
    isempty(active_chains) && return ẑ, logpos_ẑ, ∇ẑ, p̂, η_init, log_r, st_lux

    geq_bool = log_r .>= log_b

    while !isempty(active_chains)
        
        η_init[active_chains] .=
            safe_step_size_update(η_init[active_chains], δ[active_chains], Δη)

        x_active = model.lkhood.SEQ ? x[:, :, active_chains] : x[:, :, :, active_chains]

        ẑ_active, logpos_ẑ_active, ∇ẑ_active, p̂_active, log_r_active, st_lux = leapfrog(
            z[:, :, active_chains],
            ∇z[:, :, active_chains],
            x_active,
            temps[active_chains],
            logpos_z[active_chains],
            momentum[:, :, active_chains],
            M,
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
    return ẑ, logpos_ẑ, ∇ẑ, p̂, η_init, log_r, st_lux
end

function autoMALA_step(
    log_a::AbstractArray{U,1},
    log_b::AbstractArray{U,1},
    z::AbstractArray{U,3},
    ∇z::AbstractArray{U,3},
    x::AbstractArray{T},
    temps::AbstractArray{T,1},
    logpos_z::AbstractArray{U,1},
    momentum::AbstractArray{U,3},
    M::AbstractArray{U,2},
    model::T_KAM{T,U},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    η_init::AbstractArray{U,1},
    Δη::U,
    η_min::U,
    η_max::U,
    ε::U,
)::Tuple{
    AbstractArray{U,3},
    AbstractArray{U,1},
    AbstractArray{U,1},
    AbstractArray{U,1},
    AbstractArray{U,1},
    NamedTuple,
} where {T<:half_quant,U<:full_quant}

    ẑ, logpos_ẑ, ∇ẑ, p̂, η, log_r, st_lux = select_step_size(
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
    )

    z_rev, _, _, _, η_prime, _, st_lux = select_step_size(
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
    )

    reversible = check_reversibility(z, z_rev, η, η_prime; tol = ε)
    return ẑ, η, η_prime, reversible, log_r, st_lux
end

end
