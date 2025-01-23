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
    IS_samples::Int
    verbose::Bool
    p::AbstractArray{quant}
    N_t::Int
    MALA::Bool
    init_η::quant
    posterior_sample::Function
    loss_fcn::Function
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
    x̂, seed = generate_from_z(model.lkhood, ps.gen, st.gen, z; seed=seed, noise=false)
    return x̂, seed
end

function importance_loss(
    m::T_KAM,
    ps,
    st,
    x::AbstractArray{quant};
    seed::Int=1
    )
    """MLE loss with importance sampling."""
    
    # Expected prior, (if contrastive divergence)
    z, seed = m.prior.sample_z(m.prior, m.IS_samples, ps.ebm, st.ebm, seed)
    ex_prior = (m.prior.contrastive_div ? 
        mean(log_prior(m.prior, z, ps.ebm, st.ebm; normalize=m.prior.contrastive_div)) : quant(0)  
    )

    logprior = log_prior(m.prior, z, ps.ebm, st.ebm; normalize=m.prior.contrastive_div)
    x̂, seed = generate_from_z(m.lkhood, ps.gen, st.gen, z; seed=seed)
    logllhood = m.lkhood.log_lkhood_model(x, x̂)

    # Weights and resampling
    weights = @ignore_derivatives softmax(logllhood, dims=2)
    resampled_idxs, seed = m.lkhood.resample_z(weights, seed)
    weights_resampled = reduce(vcat, map(b -> weights[b:b, resampled_idxs[b, :]], 1:size(x, 2)))
    logprior_resampled = reduce(hcat, map(b -> logprior[resampled_idxs[b, :], :], 1:size(x, 2)))'
    logllhood_resampled = reduce(vcat, map(b -> logllhood[b:b, resampled_idxs[b, :]], 1:size(x, 2)))

    # Expected posterior
    @tullio loss_prior[b] := weights_resampled[b, s] * (logprior_resampled[b, s])
    loss_prior = loss_prior .- ex_prior
    @tullio loss_llhood[b] := weights_resampled[b, s] * logllhood_resampled[b, s]
    return -mean(loss_prior .+ loss_llhood), seed
end

function MALA_loss(
    m::T_KAM,
    ps,
    st,
    x::AbstractArray{quant};
    seed::Int=1
    )
    """MLE loss with MALA."""

    # MALA sampling
    z, seed = m.posterior_sample(m, x, ps, st, seed)
    ex_prior = (m.prior.contrastive_div ? 
    mean(log_prior(m.prior, z[1, :, :], ps.ebm, st.ebm; normalize=m.prior.contrastive_div)) : quant(0)  
    )
    
    # Log-dists
    logprior = log_prior(m.prior, z[2, :, :], ps.ebm, st.ebm; normalize=m.prior.contrastive_div)'
    x̂, seed = generate_from_z(m.lkhood, ps.gen, st.gen, z[2, :, :]; seed=seed)
    logllhood = m.lkhood.log_lkhood_model(x, x̂)

    # Expected posterior
    return mean(logprior .+ logllhood) - ex_prior, seed
end 

function thermo_loss(
    m::T_KAM,
    ps,
    st,
    x::AbstractArray{quant};
    seed::Int=1
    )
    """Thermodynamic Integration loss with Steppingstone sampling."""

    # Schedule temperatures, and Parallel Tempering
    temperatures = @ignore_derivatives collect(quant, [(k / m.N_t)^m.p[st.train_idx] for k in 0:m.N_t]) 
    z, seed, st = m.posterior_sample(m, x, temperatures[2:end-1], ps, st, seed) 
    temperatures = device(temperatures)
    Δt = temperatures[2:end] - temperatures[1:end-1]

    T, B, Q = size(z)
    loss = mean(log_prior(m.prior, z[1, :, :], ps.ebm, st.ebm))

    z = reshape(z, B*(T-1), Q)
    x̂, seed = generate_from_z(m.lkhood, ps.gen, st.gen, z; seed=seed)
    logllhood = m.lkhood.log_lkhood_model_tempered(x, reshape(x̂, T-1, B, Q))
    logllhood = Δt .* logllhood
    weights = @ignore_derivatives softmax(logllhood, dims=3)

    loss += sum(weights .* logllhood)
    return -loss / B, seed
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

function init_T_KAM(
    dataset::AbstractArray{quant},
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
    loss_fcn = importance_loss

    use_MALA = parse(Bool, retrieve(conf, "MALA", "use_langevin"))
    initial_step_size = parse(quant, retrieve(conf, "MALA", "initial_step_size"))
    num_steps = parse(Int, retrieve(conf, "MALA", "iters"))
    N_unadjusted = parse(Int, retrieve(conf, "MALA", "N_unadjusted"))
        
    # Importance sampling or MALA
    posterior_fcn = (m, x, ps, st, seed) -> (m.prior.sample_z(m.prior, IS_samples, ps.ebm, st.ebm, seed)..., st)
    if use_MALA && !(N_t > 1) 
        @reset prior_model.contrastive_div = true
        num_steps = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "N_langevin_per_temp"))
        posterior_fcn = (m, x, ps, st, seed) -> @ignore_derivatives autoMALA_sampler(m, ps, st, x; N=num_steps, η_init=initial_step_size, N_unadjusted=N_unadjusted, seed=seed)
        loss_fcn = MALA_loss
    end
    
    p = [quant(1)]
    if N_t > 1
        posterior_fcn = (m, x, t, ps, st, seed) -> @ignore_derivatives autoMALA_sampler(m, ps, st, x; t=t, N=num_steps, η_init=initial_step_size, N_unadjusted=N_unadjusted, seed=seed)
        @reset prior_model.contrastive_div = true
        loss_fcn = thermo_loss

        # Cyclic p schedule
        initial_p = parse(quant, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "p_start"))
        end_p = parse(quant, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "p_end"))
        num_cycles = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_cycles"))
        num_param_updates = parse(Int, retrieve(conf, "TRAINING", "N_epochs")) * length(train_loader)
        
        x = range(0, stop=2*π*num_cycles, length=num_param_updates)
        p = initial_p .+ (end_p - initial_p) .* 0.5 .* (1 .- cos.(x)) .|> quant
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
            loss_fcn
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
        gen = Lux.initialstates(rng, model.lkhood),
        train_idx = 1
        )
end

end