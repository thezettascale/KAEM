module PopulationXchange

export ReplicaXchange, NoExchange

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

struct ReplicaXchange
    Q::Int
    P::Int
    S::Int
    num_temps::Int
end

function (r::ReplicaXchange)(
        i,
        z_i,
        x,
        temps,
        model,
        lkhood_copy,
        ps,
        st_kan,
        st_lux,
        log_u_swap,
        ll_noise,
        mask_swap_1,
        mask_swap_2,
    )
    Q, P, S, num_temps = r.Q, r.P, r.S, r.num_temps
    z = reshape(z_i, Q, P, S, num_temps)

    mask1 = mask_swap_1[:, i]
    mask2 = mask_swap_2[:, i]
    mask1_expanded = reshape(mask1, 1, 1, 1, num_temps)
    mask2_expanded = reshape(mask2, 1, 1, 1, num_temps)

    # Randomly pick two adjacent temperatures to swap
    z_t = dropdims(sum(z .* reshape(mask1, 1, 1, 1, num_temps); dims = 4); dims = 4)
    z_t1 = dropdims(sum(z .* reshape(mask2, 1, 1, 1, num_temps); dims = 4); dims = 4)

    noise_1 =
        (
        model.lkhood.SEQ ?
            dropdims(
                sum(
                    ll_noise[:, :, :, 1, :, i] .* mask1_expanded
                    ; dims = 4
                ); dims = 4
            ) :
            dropdims(
                sum(
                    ll_noise[:, :, :, :, 1, :, i] .* reshape(mask1, 1, 1, 1, 1, num_temps)
                    ; dims = 5
                ); dims = 5
            )
    )
    noise_2 =
        (
        model.lkhood.SEQ ?
            dropdims(
                sum(
                    ll_noise[:, :, :, 2, :, i] .* mask2_expanded
                    ; dims = 4
                ); dims = 4
            ) :
            dropdims(
                sum(
                    ll_noise[:, :, :, :, 2, :, i] .* reshape(mask2, 1, 1, 1, 1, num_temps)
                    ; dims = 5
                ); dims = 5
            )
    )

    ll_t = first(
        log_likelihood_MALA(
            z_t,
            x,
            lkhood_copy,
            ps.gen,
            st_kan.gen,
            st_lux.gen,
            noise_1;
            ε = model.ε,
        )
    )
    ll_t1 = first(
        log_likelihood_MALA(
            z_t1,
            x,
            lkhood_copy,
            ps.gen,
            st_kan.gen,
            st_lux.gen,
            noise_2;
            ε = model.ε,
        )
    )

    # Global exchange criterion
    temps_t = sum(temps .* mask1)
    temps_t1 = sum(temps .* mask2)
    log_swap_ratio = (temps_t1 - temps_t) .* (sum(ll_t) - sum(ll_t1))
    swap = sum(log_u_swap[:, i] .* mask1) < log_swap_ratio

    z = (
        mask1_expanded .* # Index of t
            (swap .* z_t1 .+ (1 .- swap) .* z_t) .+ # Swap or not
            (1 .- mask1_expanded) .* z # Keep the rest
    )
    z = (
        mask2_expanded .* # Index of t1
            ((1 .- swap) .* z_t1 .+ swap .* z_t) .+ # Swap or not
            (1 .- mask2_expanded) .* z # Keep the rest
    )

    return reshape(z, Q, P, S * num_temps)
end

struct NoExchange end

function (r::NoExchange)(
        i,
        z_i,
        x,
        temps,
        model,
        lkhood_copy,
        ps,
        st_kan,
        st_lux,
        log_u_swap,
        ll_noise,
        mask_swap_1,
        mask_swap_2,
    )
    return z_i
end

end
