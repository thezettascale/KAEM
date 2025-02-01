module T_KAM_model

export T_KAM, init_T_KAM, generate_batch, update_model_grid

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Accessors, ComponentArrays, Statistics, LuxCUDA
using Flux: DataLoader
using NNlib: sigmoid_fast
using ChainRules: @ignore_derivatives
using Zygote: Buffer

include("mixture_prior.jl")
include("KAN_likelihood.jl")
include("langevin_sampling.jl")
include("univariate_functions.jl")
include("../utils.jl")
using .ebm_mix_prior
using .KAN_likelihood
using .LangevinSampling: autoMALA_sampler
using .univariate_functions: update_fcn_grid, fwd
using .Utils: device, next_rng, half_quant, full_quant

struct T_KAM <: Lux.AbstractLuxLayer
    prior::mix_prior
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

    z, seed = model.prior.sample_z(model.prior, num_samples, ps.ebm, st.ebm, seed)
    x̂, _ = model.lkhood.generate_from_z(model.lkhood, ps.gen, Lux.testmode(st.gen), z)
    return model.lkhood.output_activation(x̂), seed
end

function importance_loss(
    m::T_KAM,
    ps,
    st,
    x::AbstractArray{half_quant};
    full_precision::Bool=false, # Switches to full precision for accumulation over samples - set to false when AD is used or else LLVM will crash
    seed::Int=1
    )
    """MLE loss with importance sampling."""
    
    # Expected prior, (if contrastive divergence)
    z, seed = m.prior.sample_z(m.prior, m.IS_samples, ps.ebm, st.ebm, seed)
    ex_prior = full_precision ? full_quant(0) : half_quant(0)
    ex_prior = (m.prior.contrastive_div ? 
    mean(log_prior(m.prior, z, ps.ebm, st.ebm; normalize=m.prior.contrastive_div, full_precision=full_precision, ε=m.ε)) : ex_prior
    )

    logprior = log_prior(m.prior, z, ps.ebm, st.ebm; normalize=m.prior.contrastive_div, full_precision=full_precision, ε=m.ε)
    logllhood, st_gen, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z; full_precision=full_precision, seed=seed, ε=m.ε)
    @ignore_derivatives @reset st.gen = st_gen

    # Weights and resampling
    weights = @ignore_derivatives softmax(full_quant.(logllhood), dims=2) 
    resampled_idxs, seed = m.lkhood.resample_z(weights, seed)
    weights_resampled = @ignore_derivatives reduce(vcat, map(b -> weights[b:b, resampled_idxs[b, :]], 1:size(x)[end])) 
    logprior_resampled = reduce(hcat, map(b -> logprior[resampled_idxs[b, :], :], 1:size(x)[end]))'
    logllhood_resampled = reduce(vcat, map(b -> logllhood[b:b, resampled_idxs[b, :]], 1:size(x)[end]))

    # Expected posterior
    logprior_resampled = logprior_resampled .- ex_prior
    @tullio loss_prior[b] := weights_resampled[b, s] * (logprior_resampled[b, s])
    @tullio loss_llhood[b] := weights_resampled[b, s] * logllhood_resampled[b, s]

    m.verbose && println("Prior loss: ", -mean(loss_prior), " LLhood loss: ", -mean(loss_llhood))
    return -mean(loss_prior .+ loss_llhood)*m.loss_scaling, st, seed
end

function MALA_loss(
    m::T_KAM,
    ps,
    st,
    x::AbstractArray{half_quant};
    full_precision::Bool=false, # Switches to full precision for accumulation over samples - set to false when AD is used or else LLVM will crash
    seed::Int=1
    )
    """MLE loss with MALA."""

    # MALA sampling
    z, st, seed = m.posterior_sample(m, x, ps, st, seed)
    ex_prior = full_precision ? full_quant(0) : half_quant(0)
    ex_prior = (m.prior.contrastive_div ? 
    mean(log_prior(m.prior, z[1, :, :], ps.ebm, st.ebm; normalize=m.prior.contrastive_div, full_precision=full_precision, ε=m.ε)) : ex_prior
    )
    
    # Log-dists
    logprior = log_prior(m.prior, z[2, :, :], ps.ebm, st.ebm; normalize=m.prior.contrastive_div, full_precision=full_precision, ε=m.ε)'
    logllhood, st_gen, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z[2, :, :]; full_precision=full_precision, seed=seed, ε=m.ε)
    @ignore_derivatives @reset st.gen = st_gen

    # Expected posterior
    m.verbose && println("Prior loss: ", -mean(logprior .- ex_prior), " LLhood loss: ", -mean(logllhood))
    return -mean(logprior .- ex_prior .+ logllhood)*m.loss_scaling, st, seed
end 

function thermo_loss(
    m::T_KAM,
    ps,
    st,
    x::AbstractArray{half_quant};
    full_precision::Bool=false, # Switches to full precision for accumulation over samples - set to false when AD is used or else LLVM will crash
    seed::Int=1
    )
    """Thermodynamic Integration loss with Steppingstone sampling."""

    # Schedule temperatures, and Parallel Tempering
    temperatures = @ignore_derivatives collect(full_quant, [(k / m.N_t)^m.p[st.train_idx] for k in 0:m.N_t]) 
    z, st, seed = m.posterior_sample(m, x, temperatures[2:end-1], ps, st, seed) 
    temperatures = device(temperatures)
    Δt = temperatures[2:end] - temperatures[1:end-1]

    T, S, Q, B = size(z)..., size(x)[end]
    ex_prior = full_precision ? full_quant(0) : half_quant(0)
    ex_prior = (m.prior.contrastive_div ? 
    mean(log_prior(m.prior, z[1, :, :], ps.ebm, st.ebm; normalize=m.prior.contrastive_div, full_precision=full_precision, ε=m.ε)) : ex_prior
    )

    logprior = log_prior(m.prior, z[end, :, :], ps.ebm, st.ebm; normalize=m.prior.contrastive_div, full_precision=full_precision, ε=m.ε)'
    logllhood, st_gen, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, reshape(z, S*T, Q); full_precision=full_precision, seed=seed, ε=m.ε)
    @ignore_derivatives @reset st.gen = st_gen

    logllhood = reshape(logllhood, T, B, S)
    logllhood = Δt .* logllhood
    weights = @ignore_derivatives softmax(full_quant.(logllhood), dims=3) 

    # Expected posterior
    TI_loss = sum(weights .* logllhood)
    MLE_loss = sum(sum(weights[end, :, :] .* (logprior .- ex_prior .+ logllhood[end, :, :]); dims=2))
    
    m.verbose && println("Prior loss: ", -mean(logprior .- ex_prior), " LLhood loss: ", -mean(logllhood[end, :, :]))
    return -((TI_loss + MLE_loss) / 2B)*m.loss_scaling, st, seed
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
        z, seed = model.prior.sample_z(model.prior, model.grid_updates_samples, ps.ebm, st.ebm, seed)

        for i in 1:model.prior.depth
            new_grid, new_coef = update_fcn_grid(model.prior.fcns_qp[Symbol("$i")], ps.ebm[Symbol("$i")], st.ebm[Symbol("$i")], z)
            @reset ps.ebm[Symbol("$i")].coef = new_coef
            @reset model.prior.fcns_qp[Symbol("$i")].grid = new_grid

            z = fwd(model.prior.fcns_qp[Symbol("$i")], ps.ebm[Symbol("$i")], st.ebm[Symbol("$i")], z)
            z = i == 1 ? reshape(z, :, size(z, 3)) : dropdims(sum(z, dims=2); dims=2)
        end
    end
         
    (!model.update_llhood_grid || model.lkhood.CNN) && return model, ps, seed

    z, seed = model.prior.sample_z(model.prior, model.grid_updates_samples, ps.ebm, st.ebm, seed)

    for i in 1:model.lkhood.depth
        new_grid, new_coef = update_fcn_grid(model.lkhood.Φ_fcns[Symbol("$i")], ps.gen[Symbol("$i")], st.gen[Symbol("$i")], z)
        @reset ps.gen[Symbol("$i")].coef = new_coef
        @reset model.lkhood.Φ_fcns[Symbol("$i")].grid = new_grid

        z = fwd(model.lkhood.Φ_fcns[Symbol("$i")], ps.gen[Symbol("$i")], st.gen[Symbol("$i")], z)
        z = dropdims(sum(z, dims=2); dims=2)
    end

    return model, full_quant.(ps), seed
end

function init_T_KAM(
    dataset::AbstractArray{half_quant},
    conf::ConfParse;
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
    seq = parse(Int, retrieve(conf, "KAN_LIKELIHOOD", "sequence_length")) > 0

    data_seed, rng = next_rng(data_seed)
    train_data = (
        cnn ? dataset[:,:,:,1:N_train] : 
        (seq ? dataset[:, :, 1:N_train] : dataset[:, 1:N_train]) 
        ) |> device

    test_data = (
        cnn ? dataset[:,:,:,N_train+1:N_train+N_test] : 
        (seq ? dataset[:, :, N_train+1:N_train+N_test] : dataset[:, N_train+1:N_train+N_test]) 
        ) |> device



    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true, rng=rng)
    test_loader = DataLoader(test_data, batchsize=batch_size, shuffle=false)
    loss_scaling = parse(half_quant, retrieve(conf, "MIXED_PRECISION", "loss_scaling"))
    out_dim = cnn ? size(dataset, 3) : size(dataset, 1)
    
    prior_model = init_mix_prior(conf; prior_seed=prior_seed)
    lkhood_model = init_KAN_lkhood(conf, out_dim; lkhood_seed=lkhood_seed)

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
    posterior_fcn = (m, x, ps, st, seed) -> (m.prior.sample_z(m.prior, IS_samples, ps.ebm, st.ebm, seed)..., st)
    if use_MALA && !(N_t > 1) 
        posterior_fcn = (m, x, ps, st, seed) -> @ignore_derivatives autoMALA_sampler(m, ps, st, x; N=num_steps, N_unadjusted=N_unadjusted, Δη=Δη, η_min=η_minmax[1], η_max=η_minmax[2], seed=seed)
        loss_fcn = MALA_loss
    end
    
    p = [full_quant(1)]
    if N_t > 1
        num_steps = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "N_langevin_per_temp"))
        posterior_fcn = (m, x, t, ps, st, seed) -> @ignore_derivatives autoMALA_sampler(m, ps, st, x; t=t, N=num_steps, N_unadjusted=N_unadjusted, Δη=Δη, η_min=η_minmax[1], η_max=η_minmax[2], seed=seed)
        loss_fcn = thermo_loss

        # Cyclic p schedule
        initial_p = parse(full_quant, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "p_start"))
        end_p = parse(full_quant, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "p_end"))
        num_cycles = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_cycles"))
        num_param_updates = parse(Int, retrieve(conf, "TRAINING", "N_epochs")) * length(train_loader)
        
        x = range(0, stop=2*π*num_cycles, length=num_param_updates)
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
            eps
        )
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
        η_init = model.N_t > 1 ? repeat([model.η_init], model.N_t-1) : [model.η_init],
        train_idx = 1
        )
end

end