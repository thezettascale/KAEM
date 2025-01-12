module LV_KAM_model

export LV_KAM, init_LV_KAM, generate_batch, MLE_loss, update_llhood_grid

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Accessors, ComponentArrays, Statistics, LuxCUDA
using Flux: DataLoader
using NNlib: sigmoid_fast
using ChainRules: @ignore_derivatives

include("mixture_prior.jl")
include("KAN_likelihood.jl")
include("univariate_functions.jl")
include("../utils.jl")
using .ebm_mix_prior
using .KAN_likelihood
using .univariate_functions: update_fcn_grid, fwd
using .Utils: device, next_rng, quant

struct LV_KAM <: Lux.AbstractLuxLayer
    prior::mix_prior
    lkhood::KAN_lkhood 
    train_loader::DataLoader
    test_loader::DataLoader
    update_prior_grid::Bool
    update_llhood_grid::Bool
    grid_update_decay::quant
    grid_updates_samples::Int
    MC_samples::Int
    verbose::Bool
    temperatures::AbstractArray{quant}
    Δt::AbstractArray{quant}
end

function init_LV_KAM(
    dataset::AbstractArray,
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
    if N_t > 1
        p = parse(quant, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "p"))
        temperatures = collect(quant, [(k / N_t)^p for k in 0:N_t]) |> device
        Δt = temperatures[2:end] .- temperatures[1:end-1]
    end

    verbose && println("Using $(Threads.nthreads()) threads.")

    return LV_KAM(
            prior_model,
            lkhood_model,
            train_loader,
            test_loader,
            update_prior_grid,
            update_llhood_grid,
            grid_update_decay,
            num_grid_updating_samples,
            MC_samples,
            verbose,
            temperatures[1:end-1, :, :],
            Δt[:,:,:]
        )
end

function Lux.initialparameters(rng::AbstractRNG, model::LV_KAM)
    return (
        ebm = Lux.initialparameters(rng, model.prior), 
        gen = Lux.initialparameters(rng, model.lkhood)
        )
end

function Lux.initialstates(rng::AbstractRNG, model::LV_KAM)
    return (
        ebm = Lux.initialstates(rng, model.prior), 
        gen = Lux.initialstates(rng, model.lkhood)
        )
end

function generate_batch(
    model::LV_KAM, 
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
    m::LV_KAM, 
    ps, 
    st, 
    x::AbstractArray;
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

    z, seed = m.prior.sample_z(m.prior, m.MC_samples, ps.ebm, st.ebm, seed)
    logprior = log_prior(m.prior, z, ps.ebm, st.ebm)
    logllhood, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z; seed=seed)
    ex_prior = m.prior.contrastive_div ? mean(logprior) : quant(0)     

    ### MLE loss is default ###
    if length(m.temperatures) <= 1
        weights = @ignore_derivatives softmax(logllhood, dims=2) 
        loss_prior = weights * (logprior .- ex_prior)
        @tullio loss_llhood[b] := weights[b, s] * logllhood[b, s]
        return -mean(loss_prior .+ loss_llhood), seed
    end

    ### Thermodynamic Integration ###

    # Prepare for broadcasting, temps and batch first for contiguous access
    logprior = permutedims(logprior, [3, 2, 1])
    logllhood = permutedims(logllhood[:,:,:], [3, 1, 2])
    
    # Resample the latent variable using systematic sampling for all adjacent power posteriors
    resampled_idx_neg, seed = @ignore_derivatives systematic_sampler(logllhood, m.temperatures[1:end-1, :, :]; seed=seed) 
    resampled_idx_pos, seed = @ignore_derivatives systematic_sampler(logllhood, m.temperatures[2:end, :, :]; seed=seed)

    # Extract adjacent samples, and find importance weights
    logprior_neg, logprior_pos = logprior[resampled_idx_neg] .- ex_prior, logprior[resampled_idx_pos] .- ex_prior
    logllhood_neg, logllhood_pos = logllhood[resampled_idx_neg], logllhood[resampled_idx_pos]
    
    weights_neg = @ignore_derivatives softmax(m.Δt[1:end-1, :, :] .* logllhood_neg, dims=3)
    weights_pos = @ignore_derivatives softmax(m.Δt[2:end, :, :] .* logllhood_pos, dims=3)

    # Weight log likelihoods by current temperature
    logllhood_neg = (m.Δt[1:end-1, :, :] .+ m.temperatures[1:end-1, :, :]) .* logllhood_neg
    logllhood_pos = (m.Δt[2:end, :, :] .+ m.temperatures[2:end, :, :]) .* logllhood_pos

    # Importance sampling for adjactent power posteriors
    @tullio loss_neg[b, t] := weights_neg[b, s, t] * (logprior_neg[b, s, t] + logllhood_neg[b, s, t])
    @tullio loss_pos[b, t] := weights_pos[b, s, t] * (logprior_pos[b, s, t] + logllhood_pos[b, s, t])
    
    # Steppingstone estimator
    return -mean(sum(loss_pos - loss_neg, dims=2) + (loss_neg[:, 1:1] .- mean(logprior))), seed
end

function update_llhood_grid(
    model::LV_KAM,
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