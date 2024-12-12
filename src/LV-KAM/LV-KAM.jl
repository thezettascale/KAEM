module LV_KAM_model

export LV_KAM, init_LV_KAM, generate_batch, MLE_loss, update_llhood_grid

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Accessors, ComponentArrays, Statistics
using Flux: DataLoader
using NNlib: sigmoid_fast

include("mixture_prior.jl")
include("MoE_likelihood.jl")
include("thermodynamic_integration.jl")
include("univariate_functions.jl")
include("../utils.jl")
using .ebm_mix_prior
using .MoE_likelihood
using .ThermodynamicIntegration
using .univariate_functions: update_fcn_grid
using .Utils

struct LV_KAM <: Lux.AbstractLuxLayer
    prior::mix_prior
    lkhood::MoE_lkhood
    train_loader::DataLoader
    test_loader::DataLoader
    grid_update_decay::Float32
    grid_updates_samples::Int
end

function init_LV_KAM(
    dataset::AbstractArray,
    conf::ConfParse;
    prior_seed::Int=1,
    lkhood_seed::Int=1,
    data_seed::Int=1,
)

    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
    N_train = parse(Int, retrieve(conf, "TRAINING", "N_train"))
    N_test = parse(Int, retrieve(conf, "TRAINING", "N_test"))
    data_seed = next_rng(data_seed)
    train_loader = DataLoader(dataset[:, 1:N_train], batchsize=batch_size, shuffle=true)
    test_loader = DataLoader(dataset[:, N_train+1:N_train+N_test], batchsize=batch_size, shuffle=false)
    out_dim = size(dataset, 1)
    
    prior_model = init_mix_prior(conf; prior_seed=prior_seed)
    
    grid_update_decay = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "grid_update_decay"))
    num_grid_updating_samples = parse(Int, retrieve(conf, "MOE_LIKELIHOOD", "num_grid_updating_samples"))

    N_t = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps"))

    if N_t > 1
        p = parse(Float32, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "p"))
        temperatures = [(k / N_t)^p for k in 0:N_t] .|> Float32 |> device
        weight_fcn = x -> softmax(view(temperatures, 1, :, 1) .* x; dims=3)
        lkhood_model = init_MoE_lkhood(conf, out_dim; lkhood_seed=lkhood_seed, weight_fcn=weight_fcn)
        
        return Thermodynamic_LV_KAM(
            prior_model,
            lkhood_model,
            train_loader,
            test_loader,
            grid_update_decay,
            num_grid_updating_samples,
            temperatures,
        )
    else
        lkhood_model = init_MoE_lkhood(conf, out_dim; lkhood_seed=lkhood_seed)

        return LV_KAM(
            prior_model,
            lkhood_model,
            train_loader,
            test_loader,
            grid_update_decay,
            num_grid_updating_samples,
        )
    end
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
    model::Union{LV_KAM, Thermodynamic_LV_KAM}, 
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
    Maximum likelihood estimation loss.

    Args:
        m: The model.
        ps: The parameters of the model.
        st: The states of the model.
        x: The batch of data.
        seed: The seed for the random number generator.

    Returns:
        The negative marginal likelihood, averaged over the batch.
    """
    z, seed = m.prior.sample_z(
        m.prior, 
        size(x, 1),
        ps.ebm,
        st.ebm,
        seed
        )

    # Compute the log-distributions for these samples, (batch_size x 1)
    logprior = log_prior(m.prior, z, ps.ebm, st.ebm)
    logllhood = log_likelihood(m.lkhood, ps.gen, st.gen, x, z; seed=seed)
    posterior_weights = m.lkhood.weight_fcn(logllhood)

    # Expectation of the logprior with respect to the posterior and prior
    ex_prior = mean(logprior)
    ex_post = mean(logprior .* posterior_weights)
    loss_prior = ex_post - ex_prior

    # Expectation of the loglikelihood with respect to the posterior
    loss_llhood = mean(logllhood .* posterior_weights)

    return -(loss_prior + loss_llhood)
end

function update_llhood_grid(
    model::Union{LV_KAM, Thermodynamic_LV_KAM},
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
    z, seed = model.prior.sample_z(model.prior, model.grid_updates_samples, ps.ebm, st.ebm, seed)
    new_grid, new_coef = update_fcn_grid(model.lkhood.fcn_q, ps.gen, st.gen, z)
    @reset ps.gen.coef = new_coef
    @reset model.lkhood.fcn_q.grid = new_grid

    return model, ps, seed
end

end