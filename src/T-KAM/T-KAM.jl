module T_KAM_model

export T_KAM, init_T_KAM, generate_batch, move_to_hq, prep_model, next_temp

using CUDA, KernelAbstractions, Enzyme
using ConfParser, Random, Lux, Accessors, ComponentArrays, Statistics, LuxCUDA
using Flux: DataLoader

include("ebm/ebm_model.jl")
include("gen/gen_model.jl")
include("loss_fcns/langevin_mle.jl")
include("loss_fcns/importance_sampling.jl")
include("loss_fcns/thermodynamic.jl")
include("../utils.jl")
include("posterior_sampling/autoMALA.jl")
include("posterior_sampling/unadjusted_langevin.jl")
using .EBM_Model
using .GeneratorModel
using .LangevinMLE
using .ImportanceSampling
using .ThermodynamicIntegration
using .autoMALA_sampling: initialize_autoMALA_sampler, autoMALA_sample
using .ULA_sampling: initialize_ULA_sampler, ULA_sample
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
    posterior_sample::Function
    loss_fcn::Function
    loss_scaling::T
    ε::T
    file_loc::AbstractString
    max_samples::Int
    MALA::Bool
    conf::ConfParse
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
    z, st_ebm = model.prior.sample_z(model, num_samples, ps, kan_st, st_lux, rng)
    x̂, st_gen =
        model.lkhood.generate_from_z(model.lkhood, ps.gen, st_lux.gen, kan_st.gen, z)
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

    if N_t > 1
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
        identity,
        identity,
        loss_scaling,
        eps,
        file_loc,
        max_samples,
        parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_langevin")),
        conf,
    )
end

function init_prior_sampler(
    model::T_KAM,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    x::AbstractArray{T},
    conf::ConfParse;
    rng::AbstractRNG = Random.default_rng(),
) where {T<:half_quant}

    if model.prior.ula
        num_steps = parse(Int, retrieve(conf, "PRIOR_LANGEVIN", "iters"))
        step_size = parse(full_quant, retrieve(conf, "PRIOR_LANGEVIN", "step_size"))
        compile_mlir = parse(Bool, retrieve(conf, "TRAINING", "MLIR_compile"))

        sampler_struct = initialize_ULA_sampler(
            ps,
            st_kan,
            Lux.testmode(st_lux),
            model,
            x;
            η = step_size,
            N = num_steps,
            num_samples = size(x)[end],
            prior_sampling_bool = true,
            compile_mlir = compile_mlir,
            rng = rng,
        )

        @reset model.prior.sample_z =
            (m, n, p, ks, ls, rng) ->
                sample(sampler_struct, m, p, ks, Lux.testmode(ls), x; rng = rng)

        println("Prior sampler: ULA")
    else
        println("Prior sampler: ITS")
    end

    return model
end

## Must be called after init_prior_sampler
function init_posterior_sampler(
    model::T_KAM,
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    x::AbstractArray{T},
    conf::ConfParse;
    rng::AbstractRNG = Random.default_rng(),
) where {T<:half_quant}

    # MLE or Thermodynamic Integration
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

    # Importance sampling or MALA
    autoMALA_bool = parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_autoMALA"))
    compile_mlir = parse(Bool, retrieve(conf, "TRAINING", "MLIR_compile"))

    if (model.MALA && !(model.N_t > 1)) || model.prior.ula
        sampler_struct =
            autoMALA_bool ?
            initialize_autoMALA_sampler(
                ps,
                st_kan,
                Lux.testmode(st_lux),
                model,
                x;
                N = num_steps,
                N_unadjusted = N_unadjusted,
                η = η_init,
                Δη = Δη,
                η_min = η_minmax[1],
                η_max = η_minmax[2],
                seq = model.lkhood.seq_length > 1,
                RE_frequency = replica_exchange_frequency,
                compile_mlir = compile_mlir,
                rng = rng,
            ) :
            initialize_ULA_sampler(
                ps,
                st_kan,
                Lux.testmode(st_lux),
                model,
                x;
                η = η_init,
                N = num_steps,
                num_samples = size(x)[end],
                RE_frequency = replica_exchange_frequency,
                compile_mlir = compile_mlir,
                rng = rng,
            )
        sample_function = autoMALA_bool ? autoMALA_sample : ULA_sample


        @reset model.posterior_sample =
            (m, x, ps, ks, ls, rng) ->
                sample_function(sampler_struct, m, ps, ks, Lux.testmode(ls), x; rng = rng)

        loss_struct = initialize_langevin_loss(
            ps,
            st_kan,
            st_lux,
            model,
            x;
            compile_mlir = compile_mlir,
            rng = rng,
        )

        @reset model.loss_fcn =
            (p, ∇, ks, ls, m, x_i, train_idx, rng) -> langevin_loss(
                loss_struct,
                p,
                ∇,
                ks,
                ls,
                m,
                x_i;
                train_idx = train_idx,
                rng = rng,
            )

        println("Posterior sampler: $(autoMALA_bool ? "autoMALA" : "ULA")")
    elseif model.N_t > 1
        num_steps =
            parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "N_langevin_per_temp"))
        replica_exchange_frequency = parse(
            Int,
            retrieve(conf, "THERMODYNAMIC_INTEGRATION", "replica_exchange_frequency"),
        )
        temps = collect(T, [(k / model.N_t)^model.p[1] for k = 0:model.N_t])

        sampler_struct =
            autoMALA_bool ?
            initialize_autoMALA_sampler(
                ps,
                st_kan,
                Lux.testmode(st_lux),
                model,
                x;
                temps = temps[2:end],
                N = num_steps,
                N_unadjusted = N_unadjusted,
                η = η_init,
                Δη = Δη,
                η_min = η_minmax[1],
                η_max = η_minmax[2],
                seq = model.lkhood.seq_length > 1,
                RE_frequency = replica_exchange_frequency,
                compile_mlir = compile_mlir,
                rng = rng,
            ) :
            initialize_ULA_sampler(
                ps,
                st_kan,
                Lux.testmode(st_lux),
                model,
                x;
                η = η_init,
                temps = temps[2:end],
                N = num_steps,
                num_samples = size(x)[end],
                RE_frequency = replica_exchange_frequency,
                compile_mlir = compile_mlir,
                rng = rng,
            )
        sample_function = autoMALA_bool ? autoMALA_sample : ULA_sample


        @reset model.posterior_sample =
            (m, x, t, ps, ks, ls, rng) -> sample_function(
                sampler_struct,
                m,
                ps,
                ks,
                Lux.testmode(ls),
                x;
                temps = t,
                rng = rng,
            )

        loss_struct = initialize_thermo_loss(
            ps,
            st_kan,
            st_lux,
            model,
            x;
            compile_mlir = compile_mlir,
            rng = rng,
        )

        @reset model.loss_fcn =
            (p, ∇, ks, ls, m, x_i, train_idx, rng) -> thermodynamic_loss(
                loss_struct,
                p,
                ∇,
                ks,
                ls,
                m,
                x_i;
                train_idx = train_idx,
                rng = rng,
            )

        println("Posterior sampler: $(autoMALA_bool ? "Thermo autoMALA" : "Thermo ULA")")
    else
        loss_struct = initialize_importance_loss(
            ps,
            st_kan,
            st_lux,
            model,
            x;
            compile_mlir = compile_mlir,
            rng = rng,
        )
        @reset model.loss_fcn =
            (p, ∇, ks, ls, m, x_i, train_idx, rng) -> importance_loss(
                loss_struct,
                p,
                ∇,
                ks,
                ls,
                m,
                x_i;
                train_idx = train_idx,
                rng = rng,
            )

        println("Posterior sampler: IS")
    end

    return model
end

function move_to_hq(model::T_KAM)
    """Moves the model to half precision."""

    if model.prior.layernorm_bool
        for i = 1:(model.prior.depth-1)
            @reset model.prior.layernorms[i] = model.prior.layernorms[i] |> hq
        end
    end

    if model.lkhood.layernorm_bool
        for i = 1:(model.lkhood.depth-1)
            @reset model.lkhood.layernorms[i] = model.lkhood.layernorms[i] |> hq
        end
    end

    if model.lkhood.CNN
        for i = 1:(model.lkhood.depth-1)
            @reset model.lkhood.Φ_fcns[i] = model.lkhood.Φ_fcns[i] |> hq
            if model.lkhood.batchnorm_bool
                @reset model.lkhood.batchnorms[i] = model.lkhood.batchnorms[i] |> hq
            end
        end
        @reset model.lkhood.Φ_fcns[model.lkhood.depth] =
            model.lkhood.Φ_fcns[model.lkhood.depth] |> hq
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
    ps, st_kan, st_lux = ps |> device, st_kan |> device, st_lux |> device

    model = move_to_hq(model)
    ps_hq, st_kan = T.(ps), T.(st_kan)
    model = init_prior_sampler(model, ps_hq, st_kan, st_lux, x, model.conf; rng = rng)
    model = init_posterior_sampler(model, ps_hq, st_kan, st_lux, x, model.conf; rng = rng)
    return model, ps, st_kan, st_lux
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
