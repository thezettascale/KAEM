module T_KAM_model

export T_KAM, init_T_KAM, generate_batch, update_llhood_grid

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
using .LangevinSampling
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
    p::AbstractArray{quant}
    N_t::Int
    MALA::Bool
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
    z, seed = m.posterior_sample(m, x, ps, st, seed)
    logprior = log_prior(m.prior, z, ps.ebm, st.ebm)
    logllhood, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z; seed=seed)
    ex_prior = m.prior.contrastive_div ? mean(logprior) : quant(0)  

    weights = m.MALA ? device(ones(quant, size(logllhood))) ./ m.MC_samples : @ignore_derivatives softmax(logllhood, dims=2)
    loss_prior = weights * (logprior[:] .- ex_prior)
    @tullio loss_llhood[b] := weights[b, s] * logllhood[b, s]
    return -mean(loss_prior .+ loss_llhood), seed
end

function particle_filter_loss(
    m::T_KAM, 
    ps, 
    st, 
    x::AbstractArray{quant};
    seed::Int=1
)
    """Thermodynamic Integration loss with annealed particle filtering."""

    # Schedule temperatures
    temperatures = @ignore_derivatives collect(quant, [(k / m.N_t)^m.p[st.train_idx] for k in 1:m.N_t])

    # Parallelized on CPU after evaluating log-distributions on GPU
    iters = fld(m.num_particles, m.MC_samples)
    logprior, logllhood = zeros(quant, 1, 0), zeros(quant, size(x, 2), 0)
    for i in 1:iters
        z, seed = m.prior.sample_z(m.prior, m.MC_samples, ps.ebm, st.ebm, seed)
        logllhood_i, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z; seed=seed)

        logprior = hcat(logprior, log_prior(m.prior, z, ps.ebm, st.ebm)' |> cpu_device())
        logllhood = hcat(logllhood, logllhood_i |> cpu_device())
    end
    logprior_2, loglhood_2 = copy(logprior), copy(logllhood)
    
    # Initialize for first temperature = 0
    ex_prior = m.prior.contrastive_div ? mean(logprior) : quant(0)
    t1_resample, t2_resample = quant(0), quant(0) # Temperature at which last resample occurred
    resampled_idx1 = repeat(reshape(1:m.num_particles, 1, m.num_particles), size(x, 2), 1)
    resampled_idx2, = copy(resampled_idx1)
    
    # Particle filter at each power posterior
    KL_div = zeros(quant, size(x,2), 1)
    for t in temperatures
        resample_idx1, seed, t1_resample = m.lkhood.pf_resample(logllhood, t1_resample, t, seed)
        resample_idx2, seed, t2_resample = m.lkhood.pf_resample(loglhood_2, t2_resample, t, seed)
        logprior, logllhood = logprior[resample_idx1], logllhood[resample_idx1]
        logprior_2, loglhood_2 = logprior_2[resample_idx2], loglhood_2[resample_idx2]

        if t != quant(1) && (t1_resample == t || t2_resample == t)
            KL_div += mean(logprior .+ t.*logllhood; dims=2) - mean(logprior_2 .+ t.*loglhood_2; dims=2)
            KL_div += mean(logprior_2 .+ t.*loglhood_2; dims=2) - mean(logprior .+ t.*logllhood; dims=2)
        end
    end

    # Final particles at t=1
    loss1 = mean(logprior .+ logllhood; dims=2) .- ex_prior
    loss2 = mean(logprior_2 .+ loglhood_2; dims=2) .- ex_prior
    return -mean(loss1 + loss2 + KL_div) / 2, seed
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
    loss_fcn = importance_loss

    # Importance sampling or MALA
    use_MALA = parse(Bool, retrieve(conf, "MALA", "use_langevin"))
    posterior_fcn = (m, x, ps, st, seed) -> m.prior.sample_z(m.prior, MC_samples, ps.ebm, st.ebm, seed)
        

    if use_MALA && !(N_t > 1) # Don't even try MALA plus Thermodynamic Integration
        step_size = parse(quant, retrieve(conf, "MALA", "step_size"))
        noise_var = parse(quant, retrieve(conf, "MALA", "noise_var"))
        num_steps = parse(Int, retrieve(conf, "MALA", "iters"))

        posterior_fcn = (m, x, ps, st, seed) -> @ignore_derivatives MALA_sampler(m, ps, st, x; t=quant(1), η=step_size, σ=noise_var, N=num_steps, seed=seed)
    end
    
    num_particles = 0
    p = [quant(1)]
    if N_t > 1
        num_particles = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_particles"))
        loss_fcn = particle_filter_loss
        posterior_fcn = (m, x, t, ps, st, seed) -> @ignore_derivatives MALA_sampler(m, ps, st, x; t=t, η=step_size, σ=noise_var, N=num_steps, seed=seed)

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
            MC_samples,
            num_particles,
            verbose,
            p,
            N_t,
            use_MALA,
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