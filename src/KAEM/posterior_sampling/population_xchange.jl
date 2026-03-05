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
        x_t,
        temps,
        model,
        lkhood_copy,
        ps,
        st_kan,
        st_lux,
        log_u_swap,
        mask_swap_1,
        mask_swap_2,
        shift_down,
        shift_up,
    )
    Q, P, S, num_temps = r.Q, r.P, r.S, r.num_temps

    ll_all, _ = log_likelihood_MALA(
        z_i,
        x_t,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux.gen,
        zero(x_t);
        ε = model.ε,
    )

    ll_st = reshape(ll_all, S, num_temps)

    mask1 = reshape(mask_swap_1[:, i], 1, num_temps)
    mask2 = reshape(mask_swap_2[:, i], 1, num_temps)

    ll_shifted = ll_st * shift_down'
    temps_row = reshape(temps, 1, num_temps)
    temps_shifted = reshape(shift_down * temps, 1, num_temps)

    # Accept/reject w/ sign arithmetic (HLO compatible)
    ratio = mask1 .* (temps_row .- temps_shifted) .* (ll_shifted .- ll_st)
    log_u = log_u_swap[:, :, i]
    accept = mask1 .* max.(sign.(ratio .- log_u), 0.0f0)
    accept_upper = (accept * shift_up') .* mask2

    z = reshape(z_i, Q, P, S, num_temps)
    z_flat_temps = reshape(z, Q * P * S, num_temps)
    z_down = reshape(z_flat_temps * shift_down', Q, P, S, num_temps)
    z_up = reshape(z_flat_temps * shift_up', Q, P, S, num_temps)

    accept_exp = reshape(accept, 1, 1, S, num_temps)
    accept_upper_exp = reshape(accept_upper, 1, 1, S, num_temps)
    mask1_exp = reshape(mask1, 1, 1, 1, num_temps)
    mask2_exp = reshape(mask2, 1, 1, 1, num_temps)

    z_new = (
        mask1_exp .* (accept_exp .* z_down .+ (1.0f0 .- accept_exp) .* z) .+
            mask2_exp .* (accept_upper_exp .* z_up .+ (1.0f0 .- accept_upper_exp) .* z) .+
            (1.0f0 .- mask1_exp .- mask2_exp) .* z
    )

    return reshape(z_new, Q, P, S * num_temps)
end

struct NoExchange end

function (r::NoExchange)(
        i,
        z_i,
        x_t,
        temps,
        model,
        lkhood_copy,
        ps,
        st_kan,
        st_lux,
        log_u_swap,
        mask_swap_1,
        mask_swap_2,
        shift_down,
        shift_up,
    )
    return z_i
end

end
