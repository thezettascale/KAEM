module ULA_sampling

export initialize_ULA_sampler, ULA_sampler

using Reactant: @trace
using LinearAlgebra,
    Lux,
    Accessors,
    Statistics,
    ComponentArrays

using ..Utils
using ..KAEM_model
using ..KAEM_model.InverseTransformSampling

include("updates.jl")
using .LangevinUpdates

struct ULA_sampler
    prior_sampling_bool
    N
    η
    sqrt_2η
    model
    Q
    P
    S
    num_temps
    thermo_bool
    log_dist
end

function initialize_ULA_sampler(
        model::KAEM{T};
        η::T = 1.0f-3,
        prior_sampling_bool::Bool = false,
        N::Int = 20,
    ) where {T}

    Q, S = model.prior.q_size, model.batch_size
    P = model.prior.bool_config.mixture_model ? 1 : model.prior.p_size
    num_temps = (model.N_t > 1 && !prior_sampling_bool) ? model.N_t : 1
    thermo_bool = num_temps > 1
    log_dist = prior_sampling_bool ? unadjusted_logprior : unadjusted_logpos

    return ULA_sampler(
        prior_sampling_bool,
        N,
        η,
        sqrt(2 * η),
        model,
        Q,
        P,
        S,
        num_temps,
        thermo_bool,
        log_dist,
    )
end

function step(
        i,
        z_i,
        x,
        x_t,
        temps,
        temps_gpu,
        η,
        sqrt_2η,
        sampler,
        model,
        lkhood_copy,
        ps,
        st_kan,
        st_lux,
        noise,
        log_u_swap,
        ll_noise,
        mask_swap_1,
        mask_swap_2,
    )
    ξ = noise[:, :, :, i]
    ∇z =
        unadjusted_grad(
        z_i,
        x_t,
        temps_gpu,
        model,
        ps,
        st_kan,
        st_lux,
        sampler.log_dist,
    )
    new_z = z_i .+ η .* ∇z .+ sqrt_2η .* ξ
    return model.xchange_func(
        i,
        new_z,
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
end

function (sampler::ULA_sampler)(
        ps,
        st_kan,
        st_lux,
        x,
        st_rng;
        temps = [1.0f0],
    )
    """
    Unadjusted Langevin Algorithm (ULA) sampler to generate posterior samples.

    Args:
        m: The model.
        ps: The parameters of the model.
        st: The states of the model.
        x: The data.
        t: The temperatures if using Thermodynamic Integration.
        N: The number of iterations.
        rng: The random number generator.

        
    Unused arguments:
        N_unadjusted: The number of unadjusted iterations.
        Δη: The step size increment.
        η_min: The minimum step size.
        η_max: The maximum step size.

    Returns:
        The posterior samples.
    """
    model = sampler.model
    η = sampler.η
    sqrt_2η = sampler.sqrt_2η
    seq = model.lkhood.SEQ
    Q, P, S, num_temps = sampler.Q, sampler.P, sampler.S, sampler.num_temps
    thermo_bool = sampler.thermo_bool
    prior_sampling_bool = sampler.prior_sampling_bool

    # Initialize from prior
    z_flat = begin
        if model.prior.bool_config.ula && sampler.prior_sampling_bool
            rv = st_rng.posterior_its
            rv = model.prior.prior_type == "lognormal" ? exp.(rv) : rv
            rv
        else
            model_copy = model

            @reset model_copy.batch_size = S * num_temps
            @reset model_copy.prior.s_size = S * num_temps
            for i in 1:model_copy.prior.depth
                @reset model_copy.prior.fcns_qp[i].basis_function.S = S * num_temps
            end

            z_init, st_ebm = begin
                if model.prior.bool_config.mixture_model
                    sample_mixture(
                        model_copy.prior,
                        ps.ebm,
                        st_kan.ebm,
                        st_lux.ebm,
                        st_kan.quad,
                        st_rng;
                        ula_init = true
                    )
                else
                    sample_univariate(
                        model_copy.prior,
                        ps.ebm,
                        st_kan.ebm,
                        st_lux.ebm,
                        st_kan.quad,
                        st_rng;
                        ula_init = true
                    )
                end
            end

            z_init
        end
    end

    lkhood_copy = model.lkhood

    for i in 1:model.prior.depth
        @reset model.prior.fcns_qp[i].basis_function.S = S * num_temps
    end

    if !model.lkhood.SEQ && !model.lkhood.CNN
        for i in 1:model.lkhood.generator.depth
            @reset model.lkhood.generator.Φ_fcns[i].basis_function.S = S * num_temps
        end
    end

    @reset model.prior.s_size = S * num_temps
    @reset model.lkhood.generator.s_size = S * num_temps

    temps_gpu = repeat(temps, S)

    N_steps = sampler.N
    x_t = !prior_sampling_bool ? (
            model.lkhood.SEQ ? repeat(x, 1, 1, num_temps) :
            (model.use_pca ? repeat(x, 1, num_temps) : repeat(x, 1, 1, 1, num_temps))
        ) : nothing

    # Pre-allocate noise
    noise = st_rng.ula_noise
    log_u_swap = st_rng.log_swap
    ll_noise = st_rng.xchange_ll_noise
    swap_replica_idxs = st_rng.swap_replica_idxs

    # Traced HLO does not support int arrays, so handle mask outside
    mask_swap_1 = num_temps > 1 ? Lux.f32(1:num_temps .== swap_replica_idxs) .* 1.0f0 : nothing
    mask_swap_2 = num_temps > 1 ? Lux.f32(1:num_temps .== (swap_replica_idxs .+ 1)) .* 1.0f0 : nothing

    state = (1, z_flat)
    @trace while first(state) <= N_steps
        i, z_acc = state
        z_new = step(
            i,
            z_acc,
            x,
            x_t,
            temps,
            temps_gpu,
            η,
            sqrt_2η,
            sampler,
            model,
            lkhood_copy,
            ps,
            st_kan,
            st_lux,
            noise,
            log_u_swap,
            ll_noise,
            mask_swap_1,
            mask_swap_2,
        )
        state = (i + 1, z_new)
    end

    z = reshape(last(state), Q, P, S, num_temps)

    if prior_sampling_bool
        st_lux = st_lux.ebm
        z = dropdims(z; dims = 4)
    end

    return z, st_lux
end


end
