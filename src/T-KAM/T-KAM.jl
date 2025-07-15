module T_KAM_model

export T_KAM, init_T_KAM, generate_batch, move_to_hq

using CUDA, KernelAbstractions, Enzyme.EnzymeRules
using ConfParser, Random, Lux, Accessors, ComponentArrays, Statistics, LuxCUDA
using Flux: DataLoader

include("ebm/ebm_model.jl")
include("gen/gen_model.jl")
include("train_loss.jl")
include("../utils.jl")
include("posterior_sampling/autoMALA.jl")
include("posterior_sampling/unadjusted_langevin.jl")
using .EBM_Model
using .GeneratorModel
using .MarginalLikelihood
using .autoMALA_sampling: autoMALA_sampler
using .ULA_sampling: ULA_sampler
using .Utils: device, next_rng, half_quant, full_quant, hq

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
    MALA::Bool
    η_init::U
    posterior_sample::Function
    loss_fcn::Function
    loss_scaling::T
    ε::T
    file_loc::AbstractString
    max_samples::Int
end

function generate_batch(model::T_KAM, ps, st, num_samples::Int; seed::Int = 1)
    """
    Inference pass to generate a batch of data from the model.
    This is the same for both the standard and thermodynamic models.

    Args:
        model: The model.
        ps: The parameters of the model.
        st: The states of the model.
        seed: The seed for the random number generator.

    Returns:
        The generated data.
        The updated seed.
    """
    ps = ps .|> half_quant
    z, st_ebm, seed = model.prior.sample_z(model, num_samples, ps, Lux.testmode(st), seed)
    x̂, st_gen = model.lkhood.generate_from_z(model.lkhood, ps.gen, Lux.testmode(st.gen), z)
    return model.lkhood.output_activation(x̂), st_ebm, st_gen, seed
end

function init_T_KAM(
    dataset::AbstractArray{full_quant},
    conf::ConfParse,
    x_shape::Tuple;
    file_loc::AbstractString = "logs/",
    prior_seed::Int = 1,
    lkhood_seed::Int = 1,
    data_seed::Int = 1,
)

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

    data_seed, rng = next_rng(data_seed)
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

    prior_fcn = retrieve(conf, "EbmModel", "spline_function")
    if prior_fcn == "FFT"
        update_prior_grid = false
        commit!(conf, "EbmModel", "layer_norm", "true")
    end

    lkhood_fcn = retrieve(conf, "GeneratorModel", "spline_function")
    if lkhood_fcn == "FFT"
        update_llhood_grid = false
        commit!(conf, "GeneratorModel", "layer_norm", "true")
    end

    if prior_fcn == "Cheby"
        update_prior_grid = false
    end

    if lkhood_fcn == "Cheby" || cnn
        update_llhood_grid = false
    end

    prior_model = init_EbmModel(conf; prior_seed = prior_seed)
    lkhood_model = init_GenModel(conf, x_shape; lkhood_seed = lkhood_seed)

    if prior_model.ula
        loss_fcn = langevin_loss
        num_steps = parse(Int, retrieve(conf, "PRIOR_LANGEVIN", "iters"))
        step_size = parse(full_quant, retrieve(conf, "PRIOR_LANGEVIN", "step_size"))
        x_ = zeros(half_quant, 1, batch_size) |> device
        @reset prior_model.sample_z =
            (m, n, p, s, seed_prior) -> ULA_sampler(
                m,
                p,
                Lux.testmode(s),
                x_;
                seed = seed_prior,
                prior_η = step_size,
                ULA_prior = true,
                N = num_steps,
                num_samples = n,
            )
    end

    grid_update_decay =
        parse(half_quant, retrieve(conf, "GRID_UPDATING", "grid_update_decay"))
    num_grid_updating_samples =
        parse(Int, retrieve(conf, "GRID_UPDATING", "num_grid_updating_samples"))

    # MLE or Thermodynamic Integration
    N_t = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps"))
    loss_fcn = importance_loss

    use_MALA = parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_langevin"))
    initial_step_size =
        parse(full_quant, retrieve(conf, "POST_LANGEVIN", "initial_step_size"))
    num_steps = parse(Int, retrieve(conf, "POST_LANGEVIN", "iters"))
    N_unadjusted = parse(Int, retrieve(conf, "POST_LANGEVIN", "N_unadjusted"))
    Δη = parse(full_quant, retrieve(conf, "POST_LANGEVIN", "autoMALA_η_changerate"))
    η_minmax = parse.(full_quant, retrieve(conf, "POST_LANGEVIN", "step_size_bounds"))

    # Importance sampling or MALA
    widths = (
        try
            parse.(Int, retrieve(conf, "EbmModel", "layer_widths"))
        catch
            parse.(Int, split(retrieve(conf, "EbmModel", "layer_widths"), ","))
        end
    )
    posterior_fcn = identity
    autoMALA_bool = parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_autoMALA"))
    if (use_MALA && !(N_t > 1)) || (length(widths) > 2)
        loss_fcn = langevin_loss
        posterior_fcn =
            (m, x, t, ps, st, seed) ->
                ULA_sampler(m, ps, Lux.testmode(st), x; N = num_steps, seed = seed)
        if autoMALA_bool
            posterior_fcn =
                (m, x, t, ps, st, seed) -> autoMALA_sampler(
                    m,
                    ps,
                    Lux.testmode(st),
                    x;
                    N = num_steps,
                    N_unadjusted = N_unadjusted,
                    Δη = Δη,
                    η_min = η_minmax[1],
                    η_max = η_minmax[2],
                    seed = seed,
                )
        end
    end

    p = [one(full_quant)]
    if N_t > 1
        num_steps =
            parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "N_langevin_per_temp"))
        replica_exchange_frequency = parse(
            Int,
            retrieve(conf, "THERMODYNAMIC_INTEGRATION", "replica_exchange_frequency"),
        )

        loss_fcn = thermo_loss
        posterior_fcn =
            (m, x, t, ps, st, seed) -> ULA_sampler(
                m,
                ps,
                Lux.testmode(st),
                x;
                temps = t,
                N = num_steps,
                seed = seed,
                RE_frequency = replica_exchange_frequency,
            )
        if autoMALA_bool
            posterior_fcn =
                (m, x, t, ps, st, seed) -> autoMALA_sampler(
                    m,
                    ps,
                    Lux.testmode(st),
                    x;
                    temps = t,
                    N = num_steps,
                    N_unadjusted = N_unadjusted,
                    Δη = Δη,
                    η_min = η_minmax[1],
                    η_max = η_minmax[2],
                    seed = seed,
                    RE_frequency = replica_exchange_frequency,
                )
        end

        # Cyclic p schedule
        initial_p =
            parse(full_quant, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "p_start"))
        end_p = parse(full_quant, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "p_end"))
        num_cycles = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_cycles"))
        num_param_updates =
            parse(Int, retrieve(conf, "TRAINING", "N_epochs")) * length(train_loader)

        x = range(0, stop = 2*π*(num_cycles+0.5), length = num_param_updates+1)
        p = initial_p .+ (end_p - initial_p) .* 0.5 .* (1 .- cos.(x)) .|> full_quant
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
        use_MALA,
        initial_step_size,
        posterior_fcn,
        loss_fcn,
        loss_scaling,
        eps,
        file_loc,
        max(IS_samples, batch_size),
    )
end

function init_from_file(file_loc::AbstractString, ckpt::Int)
    """Load a model from a checkpoint file."""
    saved_data = load(file_loc * "ckpt_epoch_$ckpt.jld2")
    model = saved_data["model"] |> deepcopy
    ps = convert(ComponentArray, saved_data["params"])
    st = convert(NamedTuple, saved_data["state"])
    return model, ps, st
end


function Lux.initialparameters(rng::AbstractRNG, model::T_KAM)
    return ComponentArray(
        ebm = Lux.initialparameters(rng, model.prior),
        gen = Lux.initialparameters(rng, model.lkhood),
    )
end

function Lux.initialstates(rng::AbstractRNG, model::T_KAM)
    return (
        ebm = Lux.initialstates(rng, model.prior),
        gen = Lux.initialstates(rng, model.lkhood),
        η_init = model.N_t > 1 ? repeat([model.η_init], model.max_samples, model.N_t) :
                 fill(model.η_init, model.max_samples, 1),
        train_idx = 1,
    )
end

function move_to_hq(model::T_KAM)
    """Moves the model to half precision."""

    if model.prior.layernorm
        for i = 1:(model.prior.depth-1)
            @reset model.prior.fcns_qp[Symbol("ln_$i")] =
                model.prior.fcns_qp[Symbol("ln_$i")] |> hq
        end
    end

    if model.lkhood.layernorm
        for i = 1:(model.lkhood.depth-1)
            @reset model.lkhood.Φ_fcns[Symbol("ln_$i")] =
                model.lkhood.Φ_fcns[Symbol("ln_$i")] |> hq
        end
    end

    if model.lkhood.CNN
        for i = 1:model.lkhood.depth
            @reset model.lkhood.Φ_fcns[Symbol("$i")] =
                model.lkhood.Φ_fcns[Symbol("$i")] |> hq
            if model.lkhood.batchnorm
                @reset model.lkhood.Φ_fcns[Symbol("bn_$i")] =
                    model.lkhood.Φ_fcns[Symbol("bn_$i")] |> hq
            end
        end
        @reset model.lkhood.Φ_fcns[Symbol("$(model.lkhood.depth+1)")] =
            model.lkhood.Φ_fcns[Symbol("$(model.lkhood.depth+1)")] |> hq
    end

    return model
end

end
