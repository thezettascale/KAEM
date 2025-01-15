module T_KAM_model

export T_KAM, init_T_KAM, generate_batch, MLE_loss, update_llhood_grid

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Accessors, ComponentArrays, Statistics, LuxCUDA
using Flux: DataLoader
using NNlib: sigmoid_fast
using ChainRules: @ignore_derivatives
using Zygote: Buffer

include("mixture_prior.jl")
include("KAN_likelihood.jl")
include("univariate_functions.jl")
include("../utils.jl")
using .ebm_mix_prior
using .KAN_likelihood
using .univariate_functions: update_fcn_grid, fwd
using .Utils: device, next_rng, quant

struct T_KAM <: Lux.AbstractLuxLayer
    prior::mix_prior
    lkhood::KAN_lkhood 
    train_loader::DataLoader
    test_loader::DataLoader
    update_prior_grid::Bool
    update_llhood_grid::Bool
    grid_update_decay::quant
    grid_updates_samples::Int
    MC_samples::Int
    num_particles::Int
    verbose::Bool
    temperatures::AbstractArray{quant}
end

function init_T_KAM(
    dataset::AbstractArray{quant},
    conf::ConfParse;
    prior_seed::Int=1,
    lkhood_seed::Int=1,
    data_seed::Int=1,
)

    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
    MC_samples = parse(Int, retrieve(conf, "TRAINING", "MC_expectation_sample_size"))
    N_train = parse(Int, retrieve(conf, "TRAINING", "N_train"))
    N_test = parse(Int, retrieve(conf, "TRAINING", "N_test"))
    verbose = parse(Bool, retrieve(conf, "TRAINING", "verbose"))
    update_prior_grid = parse(Bool, retrieve(conf, "GRID_UPDATING", "update_prior_grid"))
    update_llhood_grid = parse(Bool, retrieve(conf, "GRID_UPDATING", "update_llhood_grid"))
    data_seed, rng = next_rng(data_seed)
    train_loader = DataLoader(dataset[:, 1:N_train], batchsize=batch_size, shuffle=true, rng=rng)
    test_loader = DataLoader(dataset[:, N_train+1:N_train+N_test], batchsize=batch_size, shuffle=false)
    out_dim = size(dataset, 1)
    
    prior_model = init_mix_prior(conf; prior_seed=prior_seed)
    lkhood_model = init_KAN_lkhood(conf, out_dim; lkhood_seed=lkhood_seed)

    grid_update_decay = parse(quant, retrieve(conf, "GRID_UPDATING", "grid_update_decay"))
    num_grid_updating_samples = parse(Int, retrieve(conf, "GRID_UPDATING", "num_grid_updating_samples"))

    # MLE or Thermodynamic Integration
    N_t = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps"))
    temperatures = [quant(1)]
    Δt = [quant(1)]
    num_particles = 0
    if N_t > 1
        p = parse(quant, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "p"))
        temperatures = collect(quant, [(k / N_t)^p for k in 0:N_t]) 
        num_particles = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_particles"))
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
            MC_samples,
            num_particles,
            verbose,
            temperatures,
        )
end

function Lux.initialparameters(rng::AbstractRNG, model::T_KAM)
    return (
        ebm = Lux.initialparameters(rng, model.prior), 
        gen = Lux.initialparameters(rng, model.lkhood)
        )
end

function Lux.initialstates(rng::AbstractRNG, model::T_KAM)
    return (
        ebm = Lux.initialstates(rng, model.prior), 
        gen = Lux.initialstates(rng, model.lkhood)
        )
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
    z, seed = model.prior.sample_z(model.prior, num_samples, ps.ebm, st.ebm, seed)
    x̂, seed = generate_from_z(model.lkhood, ps.gen, st.gen, z; seed=seed)
    return x̂, seed
end

function MLE_loss(
    m::T_KAM, 
    ps, 
    st, 
    x::AbstractArray{quant};
    seed::Int=1
    )
    """
    Maximum likelihood estimation loss. Map is used to
    conduct importance sampling per temperature.

    Args:
        m: The model.
        ps: The parameters of the model.
        st: The states of the model.
        x: The batch of data.
        seed: The seed for the random number generator.

    Returns:
        The negative marginal likelihood, averaged over the batch.
    """
    
    ### MLE loss is default ###
    if length(m.temperatures) <= 1
        z, seed = m.prior.sample_z(m.prior, m.MC_samples, ps.ebm, st.ebm, seed)
        logprior = log_prior(m.prior, z, ps.ebm, st.ebm)
        logllhood, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z; seed=seed)
        ex_prior = m.prior.contrastive_div ? mean(logprior) : quant(0)  

        weights = @ignore_derivatives softmax(logllhood, dims=2) 
        loss_prior = weights * (logprior[:] .- ex_prior)
        @tullio loss_llhood[b] := weights[b, s] * logllhood[b, s]
        return -mean(loss_prior .+ loss_llhood), seed
    end

    ### Thermodynamic Integration ###
    
    # Parallelized on CPU after evaluating log-distributions on GPU
    iters = fld(m.num_particles, m.MC_samples)
    logprior, logllhood = zeros(quant, 1, 0), zeros(quant, size(x, 2), 0)
    for i in 1:iters
        z, seed = m.prior.sample_z(m.prior, m.MC_samples, ps.ebm, st.ebm, seed)
        logllhood_i, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z; seed=seed)

        logprior = hcat(logprior, log_prior(m.prior, z, ps.ebm, st.ebm)' |> cpu_device())
        logllhood = hcat(logllhood, logllhood_i |> cpu_device())
    end

    logprior_neg, logprior_pos = copy(logprior), copy(logprior)
    logllhood_neg, logllhood_pos = copy(logllhood), copy(logllhood)
    
    # Initialize for first sum
    loss = zeros(quant, size(x, 2))
    resampled_idx_neg = repeat(reshape(1:m.num_particles, 1, m.num_particles), size(x, 2), 1)
    resampled_idx_pos, seed = m.lkhood.pf_resample(logllhood_pos, m.temperatures[2], seed)
    
    # Particle filter at each power posterior
    for t in eachindex(m.temperatures[1:end-2])
        
        # Extract resampled particles
        logprior_neg, logprior_pos = logprior_neg[resampled_idx_neg], logprior_pos[resampled_idx_pos]
        logllhood_neg, logllhood_pos = logllhood_neg[resampled_idx_neg], logllhood_pos[resampled_idx_pos]

        # Unchanged log-prior, (KL divergence)
        loss -= dropdims(mean(logprior_pos; dims=2) - mean(logprior_neg; dims=2); dims=2)

        # Tempered log-likelihoods, (trapezium rule)
        loss -= dropdims(mean(m.temperatures[t+1] .* logllhood_pos; dims=2) - mean(m.temperatures[t] .* logllhood_neg; dims=2); dims=2)

        # Filter particles
        resampled_idx_neg, seed = m.lkhood.pf_resample(logllhood_neg, m.temperatures[t+1], seed)
        resampled_idx_pos, seed = m.lkhood.pf_resample(logllhood_pos, m.temperatures[t+2], seed)  
    end 

    # Final importance sampling on entire population
    logprior_neg, logllhood_neg = logprior_neg[resampled_idx_neg], logllhood_neg[resampled_idx_neg]
    logprior_pos, logllhood_pos = logprior_pos[resampled_idx_pos], logllhood_pos[resampled_idx_pos]
    
    # Weights should be more or less uniform
    weights = @ignore_derivatives softmax(logllhood_pos, dims=2)
    @tullio ex_prior[b] := weights[b, s] * logprior_pos[b, s]
    @tullio ex_llhood[b] := weights[b, s] * logllhood_pos[b, s]

    loss -= ex_prior .- dropdims(mean(logprior_neg; dims=2); dims=2)
    loss -= ex_llhood .- dropdims(m.temperatures[end-1] .* mean(logllhood_neg; dims=2); dims=2)
    return mean(loss), seed
end

function update_llhood_grid(
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
    
    if model.update_prior_grid
        p_size = model.prior.fcns_qp[Symbol("1")].in_dim
        z = rand(model.prior.π_0, model.grid_updates_samples, p_size) |> device

        for i in 1:model.prior.depth
            new_grid, new_coef = update_fcn_grid(model.prior.fcns_qp[Symbol("$i")], ps.ebm[Symbol("$i")], st.ebm[Symbol("$i")], z)
            @reset ps.ebm[Symbol("$i")].coef = new_coef
            @reset model.prior.fcns_qp[Symbol("$i")].grid = new_grid

            z = fwd(model.prior.fcns_qp[Symbol("$i")], ps.ebm[Symbol("$i")], st.ebm[Symbol("$i")], z)
            z = i == 1 ? reshape(z, :, size(z, 3)) : dropdims(sum(z, dims=2); dims=2)
        end
    end
         
    !model.update_llhood_grid && return model, ps, seed

    z, seed = model.prior.sample_z(model.prior, model.grid_updates_samples, ps.ebm, st.ebm, seed)

    for i in 1:model.lkhood.depth
        new_grid, new_coef = update_fcn_grid(model.lkhood.Φ_fcns[Symbol("$i")], ps.gen[Symbol("$i")], st.gen[Symbol("$i")], z)
        @reset ps.gen[Symbol("$i")].coef = new_coef
        @reset model.lkhood.Φ_fcns[Symbol("$i")].grid = new_grid

        z = fwd(model.lkhood.Φ_fcns[Symbol("$i")], ps.gen[Symbol("$i")], st.gen[Symbol("$i")], z)
        z = dropdims(sum(z, dims=2); dims=2)
    end

    return model, ps, seed
end

end