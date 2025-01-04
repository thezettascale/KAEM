module LV_KAM_model

export LV_KAM, init_LV_KAM, generate_batch, MLE_loss, update_llhood_grid

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Accessors, ComponentArrays, Statistics, LuxCUDA
using Flux: DataLoader
using NNlib: sigmoid_fast
using ChainRules: @ignore_derivatives

include("mixture_prior.jl")
include("MoE_likelihood.jl")
include("univariate_functions.jl")
include("../utils.jl")
using .ebm_mix_prior
using .MoE_likelihood
using .univariate_functions: update_fcn_grid, fwd
using .Utils: device, next_rng

struct LV_KAM <: Lux.AbstractLuxLayer
    prior::mix_prior
    lkhood::MoE_lkhood 
    train_loader::DataLoader
    test_loader::DataLoader
    update_prior_grid::Bool
    update_llhood_grid::Bool
    grid_update_decay::Float32
    grid_updates_samples::Int
    MC_samples::Int
    verbose::Bool
    temperatures::AbstractArray{Float32}
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
    lkhood_model = init_MoE_lkhood(conf, out_dim; lkhood_seed=lkhood_seed)

    grid_update_decay = parse(Float32, retrieve(conf, "GRID_UPDATING", "grid_update_decay"))
    num_grid_updating_samples = parse(Int, retrieve(conf, "GRID_UPDATING", "num_grid_updating_samples"))

    # MLE or Thermodynamic Integration
    N_t = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps"))
    temperatures = [1f0]
    if N_t > 1
        p = parse(Float32, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "p"))
        temperatures = [(k / N_t)^p for k in 0:N_t] .|> Float32 
    end

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
            temperatures,
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
    x̂, seed = generate_from_z(model.lkhood, ps.gen, st.gen, z; seed=seed, noise=false)
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
    ex_prior = mean(logprior)

    function tempered_loss(t::Float32)
        """Returns the batched loss for a given temperature."""

        # Posterior samples, resamples are drawn per batch
        posterior_weights = @ignore_derivatives softmax(t .* logllhood, dims=2) 
        resampled_idxs, seed = m.lkhood.resample_z(posterior_weights, seed)    

        function posterior_expectation(batch_idx::Int)
            """Returns the marginal likelihood for a single sample in the batch."""
            loss_prior = mean(logprior[resampled_idxs[batch_idx]]) - ex_prior
            loss_llhood = mean(t .* logllhood[batch_idx, resampled_idxs[batch_idx]])
            return loss_llhood + loss_prior 
        end
        
        loss = reduce(vcat, map(posterior_expectation, 1:size(x, 2)))
        return loss[:,:] # Singleton dimension for fast reduction across temperatures
    end

    # MLE loss is default
    length(m.temperatures) <= 1 && return -mean(tempered_loss(1f0)), seed

    # Thermodynamic Integration
    losses = reduce(hcat, map(tempered_loss, m.temperatures))
    return -mean(sum(losses[:, 2:end] - losses[:, 1:end-1]; dims=2)), seed
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
            z = i == 1 ? reshape(z, :, size(z, 3)) : sum(z, dims=2)[:, 1, :] 
        end
    end
         
    !model.update_llhood_grid && return model, ps, seed

    z, seed = model.prior.sample_z(model.prior, model.grid_updates_samples, ps.ebm, st.ebm, seed)
    q_size = size(z, 2)

    Λ = copy(z)
    for i in 1:model.lkhood.depth
        new_grid, new_coef = update_fcn_grid(model.lkhood.Λ_fcns[Symbol("Λ_$i")], ps.gen[Symbol("Λ_$i")], st.gen[Symbol("Λ_$i")], Λ)
        @reset ps.gen[Symbol("Λ_$i")].coef = new_coef
        @reset model.lkhood.Λ_fcns[Symbol("Λ_$i")].grid = new_grid

        Λ = fwd(model.lkhood.Λ_fcns[Symbol("Λ_$i")], ps.gen[Symbol("Λ_$i")], st.gen[Symbol("Λ_$i")], Λ)
        Λ = i == 1 ? reshape(Λ, prod(size(Λ)[1:2]), size(Λ, 3)) : sum(Λ, dims=2)[:, 1, :]
    end

    return model, ps, seed
end

end