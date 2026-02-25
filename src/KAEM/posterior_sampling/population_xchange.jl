module PopulationXchange

export DEOReplicaXchange, NoExchange

using ..Utils
using ..KAEM_model

include("../gen/loglikelihoods.jl")
using .LogLikelihoods: log_likelihood_MALA

struct DEOReplicaXchange
    Q::Int
    P::Int
    S::Int
    num_temps::Int
end

function (r::DEOReplicaXchange)(
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
        ll_noise,
        mask_swap_1,
        mask_swap_2,
        shift_down,
        shift_up,
    )
    Q, P, S, num_temps = r.Q, r.P, r.S, r.num_temps
    noise_flat = (
        model.lkhood.SEQ ?
            reshape(ll_noise[:, :, :, :, i], size(ll_noise, 1), size(ll_noise, 2), S * num_temps) :
            (
                model.use_pca ?
                reshape(ll_noise[:, :, :, i], size(ll_noise, 1), S * num_temps) :
                reshape(ll_noise[:, :, :, :, :, i], size(ll_noise, 1), size(ll_noise, 2), size(ll_noise, 3), S * num_temps)
            )
    )

    # Batched likelihood for all temps
    ll_all, _ = log_likelihood_MALA(
        z_i,
        x_t,
        model.lkhood,
        ps.gen,
        st_kan.gen,
        st_lux.gen,
        noise_flat;
        ε = model.ε,
    )

    # Sum over S to get per-temp likelihoods
    ll_per_temp = dropdims(sum(reshape(ll_all, S, num_temps); dims = 1); dims = 1)

    # Prep + swap masks (single-onchip pass and @trace friendly)
    mask1 = mask_swap_1[:, i]
    mask2 = mask_swap_2[:, i]
    ll_shifted = shift_down * ll_per_temp # ll_shifted[t] = ll_per_temp[t+1]
    temps_shifted = shift_down * temps # temps_shifted[t] = temps[t+1]

    # Accept/reject w/ sign arithmetic (avoids bool-to-float issues in HLO)
    ratio = mask1 .* (temps .- temps_shifted) .* (ll_shifted .- ll_per_temp)
    log_u = log_u_swap[:, i]
    accept = mask1 .* max.(sign.(ratio .- log_u), 0.0f0)

    accept_upper = (shift_up * accept) .* mask2 # Propagate accept to upper positions

    # Shifted z
    z = reshape(z_i, Q, P, S, num_temps)
    z_flat_temps = reshape(z, Q * P * S, num_temps)
    z_down_flat = z_flat_temps * shift_down' # z_down[:, t] = z[:, t+1]
    z_up_flat = z_flat_temps * shift_up' # z_up[:, t] = z[:, t-1]
    z_down = reshape(z_down_flat, Q, P, S, num_temps)
    z_up = reshape(z_up_flat, Q, P, S, num_temps)

    # Broadcasting
    accept_exp = reshape(accept, 1, 1, 1, num_temps)
    accept_upper_exp = reshape(accept_upper, 1, 1, 1, num_temps)
    mask1_exp = reshape(mask1, 1, 1, 1, num_temps)
    mask2_exp = reshape(mask2, 1, 1, 1, num_temps)

    # Bulk swaps
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
        ll_noise,
        mask_swap_1,
        mask_swap_2,
        shift_down,
        shift_up,
    )
    return z_i
end

end
