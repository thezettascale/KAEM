module T_KAM_model

export T_KAM, init_T_KAM, generate_batch, update_model_grid, move_to_hq

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Accessors, ComponentArrays, Statistics, LuxCUDA
using Flux: DataLoader, mse
using NNlib: sigmoid_fast
using ChainRules: @ignore_derivatives
using LogExpFunctions: logsumexp


include("EBM_prior.jl")
include("KAN_likelihood.jl")
include("univariate_functions.jl")
include("../utils.jl")
include("posterior_sampling/autoMALA.jl")
include("posterior_sampling/ULA.jl")
using .ebm_ebm_prior
using .KAN_likelihood
using .autoMALA_sampling: autoMALA_sampler, cross_entropy, l2
using .ULA_sampling: ULA_sampler
using .univariate_functions: update_fcn_grid, fwd
using .Utils: device, next_rng, half_quant, full_quant, hq

struct T_KAM{T<:half_quant, U<:full_quant} <: Lux.AbstractLuxLayer
    prior::ebm_prior
    lkhood::KAN_lkhood 
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
    MALA::Bool
    η_init::U
    posterior_sample::Function
    loss_fcn::Function
    loss_scaling::T    
    ε::T
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

    z, st_ebm, seed = model.prior.sample_z(model, num_samples, ps, Lux.testmode(st), seed)
    x̂, st_gen = model.lkhood.generate_from_z(model.lkhood, ps.gen, Lux.testmode(st.gen), z)
    @reset st.ebm = st_ebm
    @reset st.gen = st_gen
    return model.lkhood.output_activation(x̂), st, seed
end

function importance_loss(
    m::T_KAM,
    ps,
    st,
    x::AbstractArray{T};
    seed::Int=1
    ) where {T<:half_quant}
    """MLE loss with importance sampling."""
    
    z, st_ebm, seed = m.prior.sample_z(m, m.IS_samples, ps, st, seed)
    @reset st.ebm = st_ebm

    # Log-dists
    logprior, st_ebm = m.prior.lp_fcn(m.prior, z, ps.ebm, st.ebm; ε=m.ε, normalize=!m.prior.contrastive_div)
    ex_prior = m.prior.contrastive_div ? mean(logprior) : zero(T)
    logllhood, st_gen, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, z; seed=seed, ε=m.ε)
    @reset st.ebm = st_ebm
    @reset st.gen = st_gen

    # Weights and resampling
    weights = @ignore_derivatives softmax(full_quant.(logllhood), dims=2) 
    resampled_idxs, seed = m.lkhood.resample_z(weights, seed)
    weights_resampled = @ignore_derivatives softmax(reduce(vcat, map(b -> weights[b:b, resampled_idxs[b, :]], 1:size(x)[end])), dims=2) .|> T
    logprior_resampled = reduce(hcat, map(b -> logprior[resampled_idxs[b, :], :], 1:size(x)[end]))
    logllhood_resampled = reduce(vcat, map(b -> logllhood[b:b, resampled_idxs[b, :]], 1:size(x)[end]))

    # Expected posterior
    @tullio loss_prior[b] := weights_resampled[b, s] * logprior_resampled[s, b]
    @tullio loss_llhood[b] := weights_resampled[b, s] * logllhood_resampled[b, s]

    m.verbose && println("Prior loss: ", -mean(loss_prior), " llhood loss: ", -mean(loss_llhood))
    return -(mean(loss_prior .+ loss_llhood) - ex_prior)*m.loss_scaling, st, seed
end

function mala_loss(
    m::T_KAM,
    ps,
    st,
    x::AbstractArray{T};
    seed::Int=1
    ) where {T<:half_quant}
    """MLE loss without importance, (used when posterior expectation = MCMC estimate)."""

    # MALA sampling
    z, st, seed = m.posterior_sample(m, x, 0, ps, st, seed)

    # Log-dists
    logprior_pos, st_ebm = m.prior.lp_fcn(m.prior, z[:, :, :, 1], ps.ebm, st.ebm; ε=m.ε, normalize=!m.prior.contrastive_div)
    ll_fn = m.lkhood.seq_length > 1 ? (y_i) -> dropdims(sum(cross_entropy(y_i, x; ε=m.ε); dims=1); dims=1) : (y_i) -> dropdims(sum(l2(y_i, x; ε=m.ε); dims=(1,2,3)); dims=(1,2,3))

    function lkhood(z_i, st_i)
        x̂, st_gen = m.lkhood.generate_from_z(m.lkhood, ps.gen, st_i, z_i)
        x̂ = m.lkhood.output_activation(x̂)
        return ll_fn(x̂) ./ (2*m.lkhood.σ_llhood^2), st_gen
    end

    logllhood, st_gen = lkhood(z[:, :, :, 1], st.gen)
    contrastive_div = mean(logprior_pos)

    if m.prior.contrastive_div
        z, st_ebm, seed = m.prior.sample_z(m, size(x)[end], ps, st, seed)
        logprior, st_ebm = m.prior.lp_fcn(m.prior, z, ps.ebm, st.ebm; ε=m.ε, normalize=!m.prior.contrastive_div)
        contrastive_div -= mean(logprior)
    end

    # Expected posterior
    @ignore_derivatives begin
        m.verbose && println("Prior loss: ", contrastive_div, " LLhood loss: ", mean(logllhood))
        @reset st.ebm = st_ebm
        @reset st.gen = st_gen  
    end 

    return -(contrastive_div + mean(logllhood))*m.loss_scaling, st, seed
end


function thermo_loss(
    m::T_KAM,
    ps,
    st,
    x::AbstractArray{T};
    seed::Int=1
    ) where {T<:half_quant}
    """Annealed importance sampling (AIS) loss."""

    @ignore_derivatives m.verbose && println("--------------------------------") # To separate logs

    # Schedule temperatures
    temps = @ignore_derivatives collect(T, [(k / m.N_t)^m.p[st.train_idx] for k in 0:m.N_t]) 
    z, st, seed = m.posterior_sample(m, x, device(temps[2:end]), ps, st, seed) 
    Δt, T_length, B = temps[2:end] - temps[1:end-1], length(temps), size(x)[end]

    log_ss = zero(T)
    ll_fn = m.lkhood.seq_length > 1 ? (y_i) -> dropdims(sum(cross_entropy(y_i, x; ε=m.ε); dims=1); dims=1) : (y_i) -> dropdims(sum(l2(y_i, x; ε=m.ε); dims=(1,2,3)); dims=(1,2,3))

    function lkhood(z_i, st_i)
        x̂, st_gen = m.lkhood.generate_from_z(m.lkhood, ps.gen, st_i, z_i)
        x̂ = m.lkhood.output_activation(x̂)
        return ll_fn(x̂) ./ (2*m.lkhood.σ_llhood^2), st_gen
    end

    # Posterior 
    for k in 1:T_length-2
        logllhood, st_gen = lkhood(view(z, :, :, :, k), st.gen)   
        log_ss += mean(logllhood) * Δt[k+1] 
        @ignore_derivatives @reset st.gen = st_gen
    end

    logprior, st_ebm = m.prior.lp_fcn(m.prior, view(z, :, :, :, T_length-1), ps.ebm, st.ebm; ε=m.ε, normalize=!m.prior.contrastive_div)
    contrastive_div = mean(logprior)

    # Prior
    z, st_ebm, seed = m.prior.sample_z(m, B, ps, st, seed)
    if m.prior.contrastive_div
        logprior, st_ebm = m.prior.lp_fcn(m.prior, z, ps.ebm, st.ebm; ε=m.ε, normalize=!m.prior.contrastive_div)
        contrastive_div -= mean(logprior)
    end

    logllhood, st_gen = lkhood(z, st.gen)
    log_ss += mean(logllhood * Δt[1]) 
    @ignore_derivatives @reset st.gen = st_gen

    loss = -(log_ss + contrastive_div) 

    @ignore_derivatives begin
        m.verbose && println("TI estimate of log p(x): ", log_ss, " Contrastive divergence: ", contrastive_div)
        @reset st.ebm = st_ebm
    end

    return loss * m.loss_scaling, st, seed
end

function update_model_grid(
    model::T_KAM,
    x::AbstractArray{T},
    ps, 
    st; 
    seed::Int=1
    )  where {T<:half_quant}
    """
    Update the grid of the likelihood model using samples from the prior.

    Args:
        model: The model.
        x: Data samples.
        ps: The parameters of the model.
        st: The states of the model.

    Returns:
        The updated model.
        The updated params.
        The updated seed.
    """
    ps = ps .|> T

    temps = model.N_t > 1 ? collect(T, [(k / model.N_t)^model.p[st.train_idx] for k in 0:model.N_t])[2:end] |> device : 0

    if model.update_prior_grid

        z, _, seed = ((model.MALA || model.N_t > 1) ? 
            model.posterior_sample(model, x, temps, ps, st, seed) : 
            model.prior.sample_z(model.prior, model.grid_updates_samples, ps.ebm, st.ebm, seed)
            )

        P, Q = size(z)[1:2]
        z = reshape(z, P, Q, :)
        B = size(z, 3)
        z = reshape(z, P, Q*B)

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
         
    (!model.update_llhood_grid || model.lkhood.CNN || model.lkhood.seq_length > 1) && return model, T.(ps), st, seed

    z, _, seed = ((model.MALA || model.N_t > 1) ? 
        model.posterior_sample(model, x, temps, ps, st, seed) : 
        model.prior.sample_z(model.prior, model.grid_updates_samples, ps.ebm, st.ebm, seed))

    z = dropdims(sum(reshape(z, size(z, 1), size(z, 2), :); dims=2); dims=2)

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

    return model, T.(ps), st, seed
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

    if prior_fcn == "Cheby" || prior_fcn == "Gottlieb"
        update_prior_grid = false
    end

    if lkhood_fcn == "Cheby" || lkhood_fcn == "Gottlieb" || cnn
        update_llhood_grid = false
    end
    
    prior_model = init_ebm_prior(conf; prior_seed=prior_seed)
    lkhood_model = init_KAN_lkhood(conf, x_shape; lkhood_seed=lkhood_seed)

    if prior_model.ula
        loss_fcn = mala_loss
        num_steps = parse(Int, retrieve(conf, "PRIOR_LANGEVIN", "iters"))
        step_size = parse(full_quant, retrieve(conf, "PRIOR_LANGEVIN", "step_size"))
        x_ = zeros(half_quant, 1, batch_size) |> device
        @reset prior_model.sample_z = (m, n, p, s, seed_prior) -> @ignore_derivatives ULA_sampler(m, p, Lux.testmode(s), x_; seed=seed_prior, prior_η=step_size, ULA_prior=true, N=num_steps, num_samples=n)
    end

    grid_update_decay = parse(half_quant, retrieve(conf, "GRID_UPDATING", "grid_update_decay"))
    num_grid_updating_samples = parse(Int, retrieve(conf, "GRID_UPDATING", "num_grid_updating_samples"))

    # MLE or Thermodynamic Integration
    N_t = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps"))
    loss_fcn = importance_loss

    use_MALA = parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_langevin"))
    initial_step_size = parse(full_quant, retrieve(conf, "POST_LANGEVIN", "initial_step_size"))
    num_steps = parse(Int, retrieve(conf, "POST_LANGEVIN", "iters"))
    N_unadjusted = parse(Int, retrieve(conf, "POST_LANGEVIN", "N_unadjusted"))
    Δη = parse(full_quant, retrieve(conf, "POST_LANGEVIN", "autoMALA_η_changerate"))
    η_minmax = parse.(full_quant, retrieve(conf, "POST_LANGEVIN", "step_size_bounds"))

    # Importance sampling or MALA
    widths = (
        try 
            parse.(Int, retrieve(conf, "EBM_PRIOR", "layer_widths"))
        catch
            parse.(Int, split(retrieve(conf, "EBM_PRIOR", "layer_widths"), ","))
        end
    )
    posterior_fcn = identity
    autoMALA_bool = parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_autoMALA"))
    if (use_MALA && !(N_t > 1)) || (length(widths) > 2)
        loss_fcn = mala_loss
        posterior_fcn = (m, x, t, ps, st, seed) -> @ignore_derivatives ULA_sampler(m, ps, Lux.testmode(st), x; N=num_steps, seed=seed)
        if autoMALA_bool
            posterior_fcn = (m, x, t, ps, st, seed) -> @ignore_derivatives autoMALA_sampler(m, ps, Lux.testmode(st), x; N=num_steps, N_unadjusted=N_unadjusted, Δη=Δη, η_min=η_minmax[1], η_max=η_minmax[2], seed=seed)
        end
    end
    
    p = [one(full_quant)]
    if N_t > 1
        num_steps = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "N_langevin_per_temp"))
        replica_exchange_frequency = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "replica_exchange_frequency"))
        
        loss_fcn = thermo_loss
        posterior_fcn = (m, x, t, ps, st, seed) -> @ignore_derivatives ULA_sampler(m, ps, Lux.testmode(st), x; temps=t, N=num_steps, seed=seed, RE_frequency=replica_exchange_frequency)
        if autoMALA_bool
            posterior_fcn = (m, x, t, ps, st, seed) -> @ignore_derivatives autoMALA_sampler(m, ps, Lux.testmode(st), x; temps=t, N=num_steps, N_unadjusted=N_unadjusted, Δη=Δη, η_min=η_minmax[1], η_max=η_minmax[2], seed=seed, RE_frequency=replica_exchange_frequency)
        end
        
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
            max(IS_samples, batch_size),
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
    gen = Lux.initialparameters(rng, model.lkhood),
    )
end

function Lux.initialstates(rng::AbstractRNG, model::T_KAM)
    return (
        ebm = Lux.initialstates(rng, model.prior), 
        gen = Lux.initialstates(rng, model.lkhood),
        η_init = model.N_t > 1 ? repeat([model.η_init], model.max_samples, model.N_t) : fill(model.η_init, model.max_samples, 1),
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
            if model.lkhood.batchnorm
                @reset model.lkhood.Φ_fcns[Symbol("bn_$i")] = model.lkhood.Φ_fcns[Symbol("bn_$i")] |> hq
            end
        end
        @reset model.lkhood.Φ_fcns[Symbol("$(model.lkhood.depth+1)")] = model.lkhood.Φ_fcns[Symbol("$(model.lkhood.depth+1)")] |> hq
    end

    return model
end

end