module T_KAM_model

export T_KAM, init_T_KAM, generate_batch, update_model_grid, move_to_hq

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Accessors, ComponentArrays, Statistics, LuxCUDA
using Flux: DataLoader, mse
using NNlib: sigmoid_fast
using ChainRules: @ignore_derivatives
using Zygote: Buffer

include("EBM_prior.jl")
include("KAN_likelihood.jl")
include("autoMALA.jl")
include("univariate_functions.jl")
include("../utils.jl")
using .ebm_ebm_prior
using .KAN_likelihood
using .LangevinSampling: langevin_sampler
using .univariate_functions: update_fcn_grid, fwd
using .Utils: device, next_rng, half_quant, full_quant, hq

struct T_KAM <: Lux.AbstractLuxLayer
    prior::ebm_prior
    lkhood::KAN_lkhood 
    train_loader::DataLoader
    test_loader::DataLoader
    update_prior_grid::Bool
    update_llhood_grid::Bool
    grid_update_decay::half_quant
    grid_updates_samples::Int
    IS_samples::Int
    verbose::Bool
    p::AbstractArray{full_quant}
    N_t::Int
    MALA::Bool
    η_init::full_quant
    posterior_sample::Function
    loss_fcn::Function
    loss_scaling::half_quant    
    ε::half_quant
    file_loc::AbstractString
    max_samples::Int
end

function generate_batch(
    model::T_KAM, 
    ps, 
    st,
    num_samples::Int; 
    seed::Int=1
    )
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
    # Reduce precision, 
    ps = ps .|> half_quant

    z, st_ebm, seed = model.prior.sample_z(model.prior, num_samples, ps.ebm, Lux.testmode(st.ebm), seed)
    x̂, st_gen = model.lkhood.generate_from_z(model.lkhood, ps.gen, Lux.testmode(st.gen), z)
    @reset st.ebm = st_ebm
    @reset st.gen = st_gen
    return model.lkhood.output_activation(x̂), st, seed
end

function importance_loss(
    m::T_KAM,
    ps,
    st,
    x::AbstractArray{half_quant};
    seed::Int=1
    )
    """MLE loss with importance sampling."""
    
    z, st_ebm, seed = m.prior.sample_z(m.prior, m.IS_samples, ps.ebm, st.ebm, seed)
    @reset st.ebm = st_ebm

    # Log-dists
    logprior, st_ebm = log_prior(m.prior, z, ps.ebm, st.ebm; ε=m.ε, normalize=!m.prior.contrastive_div)
    ex_prior = m.prior.contrastive_div ? mean(logprior) : half_quant(0)
    logllhood, st_gen, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z; seed=seed, ε=m.ε)
    @reset st.ebm = st_ebm
    @reset st.gen = st_gen

    # Weights and resampling
    weights = @ignore_derivatives softmax(full_quant.(logllhood), dims=2) 
    resampled_idxs, seed = m.lkhood.resample_z(weights, seed)
    weights_resampled = @ignore_derivatives softmax(reduce(vcat, map(b -> weights[b:b, resampled_idxs[b, :]], 1:size(x)[end])), dims=2) .|> half_quant
    logprior_resampled = reduce(hcat, map(b -> logprior[resampled_idxs[b, :], :], 1:size(x)[end]))
    logllhood_resampled = reduce(vcat, map(b -> logllhood[b:b, resampled_idxs[b, :]], 1:size(x)[end]))

    # Expected posterior
    @tullio loss_prior[b] := weights_resampled[b, s] * logprior_resampled[s, b]
    @tullio loss_llhood[b] := weights_resampled[b, s] * logllhood_resampled[b, s]

    m.verbose && println("Prior loss: ", -mean(loss_prior), " llhood loss: ", - mean(loss_llhood))
    return -(mean(loss_prior .+ loss_llhood) - ex_prior)*m.loss_scaling, st, seed
end

function POST_loss(
    m::T_KAM,
    ps,
    st,
    x::AbstractArray{half_quant};
    seed::Int=1
    )
    """MLE loss without importance, (used when posterior expectation = MCMC estimate)."""


    # MALA sampling
    z, st, seed = m.posterior_sample(m, x, ps, st, seed)
    
    # Log-dists
    logprior_prior, st_ebm = log_prior(m.prior, z[:, :, :, 1], ps.ebm, st.ebm; ε=m.ε)
    logprior_pos, st_ebm = log_prior(m.prior, z[:, :, :, 2], ps.ebm, st.ebm; ε=m.ε)

    x̂, st_gen = m.lkhood.generate_from_z(m.lkhood, ps.gen, st.gen, z[:, :, :, 2])
    logllhood = mse(x̂, x; agg=mean)
    logprior = mean(logprior_prior) - mean(logprior_pos)

    @reset st.ebm = st_ebm
    @reset st.gen = st_gen

    # Expected posterior
    m.verbose && println("Prior loss: ", logprior, " LLhood loss: ", logllhood)
    return (logprior .+ logllhood)*m.loss_scaling, st, seed
end 


function thermo_loss(
    m::T_KAM,
    ps,
    st,
    x::AbstractArray{half_quant};
    seed::Int=1
    )
    """Thermodynamic Integration loss with Steppingstone sampling."""

    @ignore_derivatives m.verbose && println("--------------------------------") # To separate logs

    # Schedule temperatures, and S-MALA
    temps = @ignore_derivatives collect(half_quant, [(k / m.N_t)^m.p[st.train_idx] for k in 0:m.N_t]) 
    z, st, seed = m.posterior_sample(m, x, temps[2:end-1], ps, st, seed) # Only sample from intermediate temps
    Q, P, S, T, B = size(z)..., size(x)[end]
    loss = half_quant(0)

    for k in 1:T
        z_t = view(z, :, :, :, k)
        t1, t2 = temps[k], temps[k+1]

        # Log-dists
        logprior, st_ebm = log_prior(m.prior, z_t, ps.ebm, st.ebm; ε=m.ε, normalize=!m.prior.contrastive_div)
        logllhood, st_gen, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z_t; seed=seed, ε=m.ε)
        @reset st.ebm = st_ebm
        @reset st.gen = st_gen

        # Importance sampling for current power posterior
        weights = softmax(t2 .* logllhood - t1 .* logllhood_old, dims=2)
        resampled_idxs, seed = m.lkhood.resample_z(weights, seed)
        weights_resampled = softmax(reduce(vcat, map(b -> weights[b:b, resampled_idxs[b, :]], 1:B)), dims=2)
        logprior_resampled = reduce(hcat, map(b -> logprior[resampled_idxs[b, :], :], 1:B)) 
        logllhood_resampled = reduce(vcat, map(b -> logllhood[b:b, resampled_idxs[b, :]], 1:B))

        # Importance sampling and Monte Carlo estimators across samples
        IS_estimate = mean(sum(weights_resampled .* (logprior_resampled' + t2 .* logllhood_resampled); dims=2))
        MC_estimate = mean(logprior + reduce(vcat, map(b -> logllhood[b:b, b], 1:B)))
        loss += IS_estimate - MC_estimate

        @ignore_derivatives m.verbose && println(
            "t1: ", t1, 
            " t2: ", t2, 
            " IS_estimate: ", IS_estimate,
            " MC_estimate: ", MC_estimate,
            " logprior: ", mean(logprior),
            " tempered logllhood: ", t2 * mean(logllhood),
            " Cumulative marginal lkhood: ", loss
            )
    end

    return -loss*m.loss_scaling, st, seed
end

function update_model_grid(
    model::T_KAM,
    ps, 
    st; 
    seed::Int=1
    )
    """
    Update the grid of the likelihood model using samples from the prior.

    Args:
        model: The model.
        ps: The parameters of the model.
        st: The states of the model.

    Returns:
        The updated model.
        The updated params.
        The updated seed.
    """
    ps = ps .|> half_quant

    if model.update_prior_grid
        z, st_ebm, seed = model.prior.sample_z(model.prior, model.grid_updates_samples, ps.ebm, st.ebm, seed)
        Q, P, B = size(z)
        z = reshape(z, P, Q*B)
        @reset st.ebm = st_ebm

        for i in 1:model.prior.depth
            new_grid, new_coef = update_fcn_grid(model.prior.fcns_qp[Symbol("$i")], ps.ebm[Symbol("$i")], st.ebm[Symbol("$i")], z)
            @reset ps.ebm[Symbol("$i")].coef = new_coef
            @reset st.ebm[Symbol("$i")].grid = new_grid

            z = fwd(model.prior.fcns_qp[Symbol("$i")], ps.ebm[Symbol("$i")], st.ebm[Symbol("$i")], z)
            z = i == 1 ? reshape(z, size(z, 2), P*Q*B) : dropdims(sum(z, dims=1); dims=1)

            if model.prior.layernorm && i < model.prior.depth
                z, st_ebm = Lux.apply(model.prior.fcns_qp[Symbol("ln_$i")], z, ps.ebm[Symbol("ln_$i")], st.ebm[Symbol("ln_$i")])
                @reset st.ebm[Symbol("ln_$i")] = st_ebm
            end
        end
    end
         
    (!model.update_llhood_grid || model.lkhood.CNN || model.lkhood.seq_length > 1) && return model, half_quant.(ps), st, seed

    z, st_ebm, seed = model.prior.sample_z(model.prior, model.grid_updates_samples, ps.ebm, st.ebm, seed)
    z = dropdims(sum(z, dims=2); dims=2)
    @reset st.ebm = st_ebm

    for i in 1:model.lkhood.depth
        new_grid, new_coef = update_fcn_grid(model.lkhood.Φ_fcns[Symbol("$i")], ps.gen[Symbol("$i")], st.gen[Symbol("$i")], z)
        @reset ps.gen[Symbol("$i")].coef = new_coef
        @reset st.gen[Symbol("$i")].grid = new_grid

        z = fwd(model.lkhood.Φ_fcns[Symbol("$i")], ps.gen[Symbol("$i")], st.gen[Symbol("$i")], z)
        z = dropdims(sum(z, dims=1); dims=1)

        if model.lkhood.layernorm && i < model.lkhood.depth
            z, st_gen = Lux.apply(model.lkhood.Φ_fcns[Symbol("ln_$i")], z, ps.gen[Symbol("ln_$i")], st.gen[Symbol("ln_$i")])
            @reset st.gen[Symbol("ln_$i")] = st_gen
        end
    end

    return model, half_quant.(ps), st, seed
end

function init_T_KAM(
    dataset::AbstractArray{full_quant},
    conf::ConfParse,
    x_shape::Tuple;
    file_loc::AbstractString="logs/",
    prior_seed::Int=1,
    lkhood_seed::Int=1,
    data_seed::Int=1,
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
    test_data = seq ? dataset[:, :, N_train+1:N_train+N_test] : dataset[:, :, :, N_train+1:N_train+N_test]

    data_seed, rng = next_rng(data_seed)
    train_loader = DataLoader(train_data .|> half_quant, batchsize=batch_size, shuffle=true, rng=rng)
    test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=false)
    loss_scaling = parse(half_quant, retrieve(conf, "MIXED_PRECISION", "loss_scaling"))
    out_dim = (
        cnn ? size(dataset, 3) :
        (seq ? size(dataset, 1) : 
        size(dataset, 1) * size(dataset, 2))
    )

    prior_fcn = retrieve(conf, "EBM_PRIOR", "spline_function")
    if prior_fcn == "FFT" 
        update_prior_grid = false
        commit!(conf, "EBM_PRIOR", "layer_norm", "true")
    end

    lkhood_fcn = retrieve(conf, "KAN_LIKELIHOOD", "spline_function")
    if lkhood_fcn == "FFT" 
        update_llhood_grid = false
        commit!(conf, "KAN_LIKELIHOOD", "layer_norm", "true")
    end
    
    prior_model = init_ebm_prior(conf; prior_seed=prior_seed)
    lkhood_model = init_KAN_lkhood(conf, x_shape; lkhood_seed=lkhood_seed)

    grid_update_decay = parse(half_quant, retrieve(conf, "GRID_UPDATING", "grid_update_decay"))
    num_grid_updating_samples = parse(Int, retrieve(conf, "GRID_UPDATING", "num_grid_updating_samples"))

    # MLE or Thermodynamic Integration
    N_t = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps"))
    loss_fcn = importance_loss

    use_MALA = parse(Bool, retrieve(conf, "MALA", "use_langevin"))
    initial_step_size = parse(full_quant, retrieve(conf, "MALA", "initial_step_size"))
    num_steps = parse(Int, retrieve(conf, "MALA", "iters"))
    N_unadjusted = parse(Int, retrieve(conf, "MALA", "N_unadjusted"))
    Δη = parse(full_quant, retrieve(conf, "MALA", "autoMALA_η_changerate"))
    η_minmax = parse.(full_quant, retrieve(conf, "MALA", "step_size_bounds"))

    # Importance sampling or MALA
    posterior_fcn = identity
    if use_MALA && !(N_t > 1) 
        posterior_fcn = (m, x, ps, st, seed) -> @ignore_derivatives langevin_sampler(m, ps, st, x; N=num_steps, N_unadjusted=N_unadjusted, Δη=Δη, η_min=η_minmax[1], η_max=η_minmax[2], seed=seed)
        loss_fcn = POST_loss
    end
    
    p = [full_quant(1)]
    if N_t > 1
        num_steps = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "N_langevin_per_temp"))
        posterior_fcn = (m, x, t, ps, st, seed) -> @ignore_derivatives langevin_sampler(m, ps, st, x; t=t, N=num_steps, N_unadjusted=N_unadjusted, Δη=Δη, η_min=η_minmax[1], η_max=η_minmax[2], seed=seed)
        loss_fcn = thermo_loss

        # Cyclic p schedule
        initial_p = parse(full_quant, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "p_start"))
        end_p = parse(full_quant, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "p_end"))
        num_cycles = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_cycles"))
        num_param_updates = parse(Int, retrieve(conf, "TRAINING", "N_epochs")) * length(train_loader)
        
        x = range(0, stop=2*π*(num_cycles+0.5), length=num_param_updates+1)
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
            max(IS_samples, batch_size)
        )
end

function init_from_file(
    file_loc::AbstractString,
    ckpt::Int
    )
    """
    Load a model from a checkpoint file.
    """
    saved_data = load(file_loc * "ckpt_epoch_$ckpt.jld2")
    model = saved_data["model"] |> deepcopy
    ps = convert(ComponentArray, saved_data["params"])
    st = convert(NamedTuple, saved_data["state"])
    return model, ps, st
end


function Lux.initialparameters(rng::AbstractRNG, model::T_KAM)
    return ComponentArray(
        ebm = Lux.initialparameters(rng, model.prior), 
        gen = Lux.initialparameters(rng, model.lkhood)
        )
end

function Lux.initialstates(rng::AbstractRNG, model::T_KAM)
    return (
        ebm = Lux.initialstates(rng, model.prior), 
        gen = Lux.initialstates(rng, model.lkhood),
        η_init = model.N_t > 1 ? repeat([model.η_init], model.N_t-1, model.max_samples) : fill(model.η_init, 1, model.max_samples),
        train_idx = 1,
        )
end

function move_to_hq(model::T_KAM)
    """Moves the model to half precision."""

    if model.prior.layernorm
        for i in 1:model.prior.depth-1
            @reset model.prior.fcns_qp[Symbol("ln_$i")] = model.prior.fcns_qp[Symbol("ln_$i")] |> hq
        end
    end

    if model.lkhood.layernorm
        for i in 1:model.lkhood.depth-1
            @reset model.lkhood.Φ_fcns[Symbol("ln_$i")] = model.lkhood.Φ_fcns[Symbol("ln_$i")] |> hq
        end
    end

    if model.lkhood.CNN
        for i in 1:model.lkhood.depth
            @reset model.lkhood.Φ_fcns[Symbol("$i")] = model.lkhood.Φ_fcns[Symbol("$i")] |> hq
            @reset model.lkhood.Φ_fcns[Symbol("bn_$i")] = model.lkhood.Φ_fcns[Symbol("bn_$i")] |> hq
        end
        @reset model.lkhood.Φ_fcns[Symbol("$(model.lkhood.depth+1)")] = model.lkhood.Φ_fcns[Symbol("$(model.lkhood.depth+1)")] |> hq
    end

    return model
end

end