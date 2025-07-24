module T_KAM_model

export T_KAM, init_T_KAM, generate_batch, move_to_hq, prep_model, next_temp

using CUDA, KernelAbstractions, Enzyme
using ConfParser, Random, Lux, Accessors, ComponentArrays, Statistics, LuxCUDA
using Flux: DataLoader

include("ebm/ebm_model.jl")
include("gen/gen_model.jl")
include("ebm/inverse_transform.jl")
include("loss_fcns/langevin_mle.jl")
include("loss_fcns/importance_sampling.jl")
include("loss_fcns/thermodynamic.jl")
include("../utils.jl")
include("posterior_sampling/autoMALA.jl")
include("posterior_sampling/unadjusted_langevin.jl")
using .EBM_Model
using .GeneratorModel
using .InverseTransformSampling: sample_mixture, sample_univariate
using .ImportanceSampling
using .LangevinMLE
using .ThermodynamicIntegration
using .autoMALA_sampling
using .ULA_sampling
using .Utils: device, half_quant, full_quant, hq

struct T_KAM{T<:half_quant,U<:full_quant} <: Lux.AbstractLuxLayer
    prior::EbmModel
    lkhood::GenModel
    train_loader::DataLoader
    test_loader::DataLoader
    update_prior_grid::Bool
    update_llhood_grid::Bool
    grid_update_decay::T
    grid_updates_samples::Int
    IS_samples::Int
    verbose::Bool
    p::AbstractArray{U}
    N_t::Int
    sample_prior::Function
    posterior_sampler::Any
    loss_fcn::Any
    loss_scaling::T
    ε::T
    file_loc::AbstractString
    max_samples::Int
    MALA::Bool
end

function generate_batch(
    model::T_KAM,
    ps::ComponentArray{T},
    kan_st::ComponentArray{T},
    st_lux::NamedTuple,
    num_samples::Int;
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{AbstractArray{T},NamedTuple,NamedTuple,Int} where {T<:half_quant}
    """
    Inference pass to generate a batch of data from the model.
    This is the same for both the standard and thermodynamic models.

    Args:
        model: The model.
        ps: The parameters of the model.
        kan_st: The states of the KAN model.
        st_lux: The states of the Lux model.
        num_samples: The number of samples to generate.
        rng: The random number generator.

    Returns:
        The generated data.
        Lux states of the prior.
        Lux states of the likelihood.
    """
    ps = ps .|> half_quant
    z, st_ebm = model.sample_prior(model, num_samples, ps, kan_st, st_lux, rng)
    x̂, st_gen = model.lkhood.generator(ps.gen, st_lux.gen, kan_st.gen, z)
    noise = model.lkhood.σ_llhood * randn(rng, size(x̂))
    return model.lkhood.output_activation(x̂ + noise), st_ebm, st_gen
end

function init_T_KAM(
    dataset::AbstractArray{full_quant},
    conf::ConfParse,
    x_shape::Tuple;
    file_loc::AbstractString = "logs/",
    rng::AbstractRNG = Random.default_rng(),
)::T_KAM

    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
    IS_samples = parse(Int, retrieve(conf, "TRAINING", "importance_sample_size"))
    N_train = parse(Int, retrieve(conf, "TRAINING", "N_train"))
    N_test = parse(Int, retrieve(conf, "TRAINING", "N_test"))
    verbose = parse(Bool, retrieve(conf, "TRAINING", "verbose"))
    eps = parse(half_quant, retrieve(conf, "TRAINING", "eps"))
    update_prior_grid = parse(Bool, retrieve(conf, "GRID_UPDATING", "update_prior_grid"))
    update_llhood_grid = parse(Bool, retrieve(conf, "GRID_UPDATING", "update_llhood_grid"))
    cnn = parse(Bool, retrieve(conf, "CNN", "use_cnn_lkhood"))
    seq = parse(Int, retrieve(conf, "SEQ", "sequence_length")) > 1

    train_data = seq ? dataset[:, :, 1:N_train] : dataset[:, :, :, 1:N_train]
    test_data =
        seq ? dataset[:, :, (N_train+1):(N_train+N_test)] :
        dataset[:, :, :, (N_train+1):(N_train+N_test)]

    train_loader = DataLoader(
        train_data .|> half_quant,
        batchsize = batch_size,
        shuffle = true,
        rng = rng,
    )
    test_loader = DataLoader(test_data, batchsize = batch_size, shuffle = false)
    loss_scaling = parse(half_quant, retrieve(conf, "MIXED_PRECISION", "loss_scaling"))
    out_dim = (
        cnn ? size(dataset, 3) :
        (seq ? size(dataset, 1) : size(dataset, 1) * size(dataset, 2))
    )

    lkhood_fcn = retrieve(conf, "GeneratorModel", "spline_function")
    if lkhood_fcn == "FFT" || lkhood_fcn == "Cheby" || cnn
        update_llhood_grid = false
    end

    prior_model = init_EbmModel(conf; rng = rng)
    lkhood_model = init_GenModel(conf, x_shape; rng = rng)

    grid_update_decay =
        parse(half_quant, retrieve(conf, "GRID_UPDATING", "grid_update_decay"))
    num_grid_updating_samples =
        parse(Int, retrieve(conf, "GRID_UPDATING", "num_grid_updating_samples"))

    max_samples = max(IS_samples, batch_size)
    η_init = parse(full_quant, retrieve(conf, "POST_LANGEVIN", "initial_step_size"))
    N_t = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps"))
    p = [one(full_quant)]

    zero_vec = device(zeros(half_quant, x_shape..., IS_samples, batch_size))
    loss_struct = ImportanceLoss(zero_vec)

    MALA = parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_langevin"))
    autoMALA_bool = parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_autoMALA"))

    if N_t > 1
        initial_p =
            parse(full_quant, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "p_start"))
        end_p = parse(full_quant, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "p_end"))
        num_cycles = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_cycles"))
        num_param_updates =
            parse(Int, retrieve(conf, "TRAINING", "N_epochs")) * length(train_loader)

        x = range(0, stop = 2*π*(num_cycles+0.5), length = num_param_updates+1)
        p = initial_p .+ (end_p - initial_p) .* 0.5 .* (1 .- cos.(x)) .|> full_quant

        loss_struct = ThermodynamicLoss()

        type = autoMALA_bool ? "Thermo autoMALA" : "Thermo ULA"
        println("Posterior sampler: $type")
    elseif MALA || prior_model.ula
        loss_struct = LangevinLoss()
        type = autoMALA_bool ? "autoMALA" : "ULA"
        println("Posterior sampler: $type")
    else
        println("Posterior sampler: IS")
    end

    sample_prior =
        (m, n, p, sk, sl, r) ->
            sample_univariate(m.prior, n, p.ebm, sk.ebm, sl.ebm; rng = r, ε = m.ε)

    if prior_model.ula
        num_steps = parse(Int, retrieve(conf, "PRIOR_LANGEVIN", "iters"))
        step_size = parse(full_quant, retrieve(conf, "PRIOR_LANGEVIN", "step_size"))

        prior_sampler = initialize_ULA_sampler(;
            η = step_size,
            N = num_steps,
            prior_sampling_bool = true,
        )

        sample_prior =
            (m, n, p, sk, sl, r) -> prior_sampler(m, p, sk, Lux.testmode(sl), x; rng = r)
        println("Prior sampler: ULA")

    elseif prior_model.mixture_model
        sample_prior =
            (m, n, p, sk, sl, r) ->
                sample_mixture(m.prior, n, p.ebm, sk.ebm, sl.ebm; rng = r, ε = m.ε)
        println("Prior sampler: Mix ITS")
    else
        println("Prior sampler: Univar ITS")
    end

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

    posterior_sampler = initialize_ULA_sampler(;
        η = η_init,
        N = num_steps,
        RE_frequency = replica_exchange_frequency,
    )

    if autoMALA_bool
        posterior_sampler = initialize_autoMALA_sampler(;
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

    verbose && println("Using $(Threads.nthreads()) threads.")

    return T_KAM(
        prior_model,
        lkhood_model,
        train_loader,
        test_loader,
        update_prior_grid,
        update_llhood_grid,
        grid_update_decay,
        num_grid_updating_samples,
        IS_samples,
        verbose,
        p,
        N_t,
        sample_prior,
        posterior_sampler,
        loss_struct,
        loss_scaling,
        eps,
        file_loc,
        max_samples,
        MALA,
    )
end

function move_to_hq(model::T_KAM)
    """Moves the model to half precision."""

    if model.prior.layernorm_bool
        for i = 1:(model.prior.depth-1)
            @reset model.prior.layernorms[i] = model.prior.layernorms[i] |> hq
        end
    end

    if model.lkhood.generator.layernorm_bool
        for i = 1:(model.lkhood.generator.depth-1)
            @reset model.lkhood.layernorms[i] = model.lkhood.layernorms[i] |> hq
        end
    end

    if model.lkhood.CNN
        for i = 1:(model.lkhood.generator.depth-1)
            @reset model.lkhood.Φ_fcns[i] = model.lkhood.Φ_fcns[i] |> hq
            if model.lkhood.generator.batchnorm_bool
                @reset model.lkhood.batchnorms[i] = model.lkhood.batchnorms[i] |> hq
            end
        end
        @reset model.lkhood.Φ_fcns[model.lkhood.generator.depth] =
            model.lkhood.Φ_fcns[model.lkhood.generator.depth] |> hq
    end

    return model
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
    model = move_to_hq(model)
    return model, ps, T.(st_kan), st_lux
end

function init_from_file(file_loc::AbstractString, ckpt::Int)
    """Load a model from a checkpoint file."""
    saved_data = load(file_loc * "ckpt_epoch_$ckpt.jld2")
    model = saved_data["model"] |> deepcopy
    ps = convert(ComponentArray, saved_data["params"])
    st_kan = convert(NamedTuple, saved_data["kan_state"])
    st_lux = convert(NamedTuple, saved_data["lux_state"])
    return model, ps, st_kan, st_lux
end

function Lux.initialparameters(
    rng::AbstractRNG,
    model::T_KAM{T,U},
) where {T<:half_quant,U<:full_quant}
    return ComponentArray(
        ebm = Lux.initialparameters(rng, model.prior),
        gen = Lux.initialparameters(rng, model.lkhood),
    )
end

function Lux.initialstates(
    rng::AbstractRNG,
    model::T_KAM{T,U},
) where {T<:half_quant,U<:full_quant}

    ebm_kan, ebm_lux = Lux.initialstates(rng, model.prior)
    gen_kan, gen_lux = Lux.initialstates(rng, model.lkhood)

    return ComponentArray(ebm = ebm_kan, gen = gen_kan), (ebm = ebm_lux, gen = gen_lux)
end

end
