module ModelSetup

export prep_model

using ConfParser, Lux, Accessors, ComponentArrays, LuxCUDA

include("T-KAM.jl")
include("ebm/log_prior_fcns.jl")
include("loss_fcns/langevin_mle.jl")
include("loss_fcns/importance_sampling.jl")
include("loss_fcns/thermodynamic.jl")
include("posterior_sampling/autoMALA.jl")
include("posterior_sampling/unadjusted_langevin.jl")
include("../utils.jl")
using .T_KAM_model: T_KAM
using .LogPriorFCNs: log_prior_ula, log_prior_mix, log_prior_univar
using .ImportanceSampling
using .LangevinMLE
using .ThermodynamicIntegration
using .autoMALA_sampling
using .ULA_sampling
using .Utils: device, half_quant, full_quant, hq

function move_to_hq!(model::T_KAM)
    """Moves the model to half precision."""

    if model.prior.layernorm_bool
        for i = 1:(model.prior.depth-1)
            @reset model.prior.layernorms[i] = model.prior.layernorms[i] |> hq
        end
    end

    if model.lkhood.generator.layernorm_bool
        for i = 1:(model.lkhood.generator.depth-1)
            @reset model.lkhood.generator.layernorms[i] =
                model.lkhood.generator.layernorms[i] |> hq
        end
    end

    if model.lkhood.CNN
        for i = 1:(model.lkhood.generator.depth-1)
            @reset model.lkhood.generator.Φ_fcns[i] = model.lkhood.generator.Φ_fcns[i] |> hq
            if model.lkhood.generator.batchnorm_bool
                @reset model.lkhood.generator.batchnorms[i] =
                    model.lkhood.generator.batchnorms[i] |> hq
            end
        end
        @reset model.lkhood.generator.Φ_fcns[model.lkhood.generator.depth] =
            model.lkhood.generator.Φ_fcns[model.lkhood.generator.depth] |> hq
    end

    return nothing
end

function setup_training!(model::T_KAM)
    conf = model.conf
    autoMALA_bool = parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_autoMALA"))

    # Posterior samplers
    initial_step_size =
        parse(full_quant, retrieve(conf, "POST_LANGEVIN", "initial_step_size"))
    num_steps = parse(Int, retrieve(conf, "POST_LANGEVIN", "iters"))
    N_unadjusted = parse(Int, retrieve(conf, "POST_LANGEVIN", "N_unadjusted"))
    η_init = parse(full_quant, retrieve(conf, "POST_LANGEVIN", "initial_step_size"))
    Δη = parse(full_quant, retrieve(conf, "POST_LANGEVIN", "autoMALA_η_changerate"))
    η_minmax = parse.(full_quant, retrieve(conf, "POST_LANGEVIN", "step_size_bounds"))
    replica_exchange_frequency = parse(
        Int,
        retrieve(conf, "THERMODYNAMIC_INTEGRATION", "replica_exchange_frequency"),
    )

    if model.N_t > 1
        @reset model.loss_fcn = ThermodynamicLoss()

        type = autoMALA_bool ? "Thermo autoMALA" : "Thermo ULA"
        println("Posterior sampler: $type")
    elseif model.MALA || model.prior.ula
        @reset model.loss_fcn = LangevinLoss()
        type = autoMALA_bool ? "autoMALA" : "ULA"
        println("Posterior sampler: $type")
    else
        @reset model.loss_fcn = ImportanceLoss()
        println("Posterior sampler: IS")
    end

    if model.prior.ula
        num_steps = parse(Int, retrieve(conf, "PRIOR_LANGEVIN", "iters"))
        step_size = parse(full_quant, retrieve(conf, "PRIOR_LANGEVIN", "step_size"))

        prior_sampler = initialize_ULA_sampler(;
            η = step_size,
            N = num_steps,
            prior_sampling_bool = true,
        )

        @reset model.sample_prior =
            (m, n, p, sk, sl, r) -> prior_sampler(m, p, sk, Lux.testmode(sl), x; rng = r)

        @reset model.prior.lp_fcn = log_prior_ula
        println("Prior sampler: ULA")
    elseif model.prior.mixture_model
        @reset model.sample_prior =
            (m, n, p, sk, sl, r) ->
                sample_mixture(m.prior, n, p.ebm, sk.ebm, sl.ebm; rng = r, ε = m.ε)

        @reset model.prior.lp_fcn = log_prior_mix
        println("Prior sampler: Mix ITS")
    else
        @reset model.prior.lp_fcn = log_prior_univar
        println("Prior sampler: Univar ITS")
    end

    if autoMALA_bool
        @reset model.posterior_sampler = initialize_autoMALA_sampler(;
            N = num_steps,
            N_unadjusted = N_unadjusted,
            η = η_init,
            Δη = Δη,
            η_min = η_minmax[1],
            η_max = η_minmax[2],
            seq = model.lkhood.SEQ,
            RE_frequency = replica_exchange_frequency,
            samples = max_samples,
        )
    end
end

function prep_model(
    model::T_KAM,
    x::AbstractArray{T};
    rng::AbstractRNG = Random.default_rng(),
) where {T<:half_quant}
    ps = Lux.initialparameters(rng, model)
    st_kan, st_lux = Lux.initialstates(rng, model)
    ps, st_kan, st_lux =
        ps |> ComponentArray |> device, st_kan |> ComponentArray |> device, st_lux |> device
    move_to_hq!(model)
    setup_training!(model)
    return model, ps, T.(st_kan), st_lux
end

end
