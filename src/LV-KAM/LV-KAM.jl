module LV_KAM_model

export LV_KAM, init_LV_KAM, generate_batch, MLE_loss, update_llhood_grid

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Accessors
using Flux: DataLoader

include("mixture_prior.jl")
include("MoE_likelihood.jl")
include("univariate_functions.jl")
include("../utils.jl")
using .ebm_mix_prior
using .MoE_likelihood
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
    test_loader = DataLoader(dataset[:, N_train+1:N_test], batchsize=batch_size, shuffle=false)
    out_dim = size(dataset, 1)
    
    prior_model = init_mix_prior(conf; prior_seed=prior_seed)
    lkhood_model = init_MoE_lkhood(conf, out_dim; lkhood_seed=lkhood_seed)
    
    grid_update_decay = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "grid_update_decay"))
    num_grid_updating_samples = parse(Int, retrieve(conf, "MOE_LIKELIHOOD", "num_grid_updating_samples"))

    return LV_KAM(
        prior_model,
        lkhood_model,
        train_loader,
        test_loader,
        grid_update_decay,
        num_grid_updating_samples,
    )
end

function Lux.initialparameters(rng::AbstractRNG, model::LV_KAM)
    return (ebm = Lux.initialparameters(rng, model.prior), gen = Lux.initialparameters(rng, model.lkhood))
end

function Lux.initialstates(rng::AbstractRNG, model::LV_KAM)
    return (ebm = Lux.initialstates(rng, model.prior), gen = Lux.initialstates(rng, model.lkhood))
end

function generate_batch(model::LV_KAM, ps, st, num_samples; seed)
    """
    Inference pass to generate a batch of data from the model.

    Args:
        model: The model.
        ps: The parameters of the model.
        st: The states of the model.
        seed: The seed for the random number generator.

    Returns:
        The generated data.
        The updated seed.
    """
    z, seed = sample_prior(model.prior, num_samples, ps.ebm, st.ebm; init_seed=seed)
    return generate_from_z(model.lkhood, ps.gen, st.gen, z; seed=seed)
end

function MLE_loss(model::LV_KAM, ps, st, x; seed=1)
    """
    Maximum likelihood estimation loss.

    Args:
        model: The model.
        ps: The parameters of the model.
        st: The states of the model.
        x: The batch of data.
        seed: The seed for the random number generator.

    Returns:
        The loss.
        The updated seed.
    """

    func = (z, p) -> log_likelihood(model.lkhood, p.gen, st.gen, x, z; seed=seed) + log_prior(model.prior, z, p.ebm, st.ebm)
    marginal_llhood, seed = expected_posterior(model.prior, model.lkhood, ps, st, x, func, ps)
    return -marginal_llhood
end

function update_llhood_grid(model::LV_KAM, ps, st; seed=1)
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
    z, seed = sample_prior(model.prior, model.grid_updates_samples, ps.ebm, st.ebm; init_seed=seed)
    new_grid, new_coef = update_fcn_grid(model.lkhood.fcn_q, ps.gen, st.gen, z)
    @reset ps.gen.coef = new_coef
    @reset model.lkhood.fcn_q.grid = new_grid

    return model, ps, seed
end

end