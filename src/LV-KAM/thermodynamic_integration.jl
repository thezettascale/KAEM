module ThermodynamicIntegration

export Thermodynamic_LV_KAM, TI_loss

using CUDA, KernelAbstractions, Tullio, LuxCUDA
using Random, Lux, Statistics, LinearAlgebra, Distributions
using Flux: DataLoader, onehotbatch
using ChainRules: @ignore_derivatives
using NNlib: softmax

# Inherit from parent module
if isdefined(Main, :LV_KAM_model)
    using Main.LV_KAM_model.ebm_mix_prior: mix_prior, log_prior
    using Main.LV_KAM_model.MoE_likelihood: MoE_lkhood, log_likelihood 
elseif isdefined(Main, :trainer) && isdefined(Main.trainer, :LV_KAM_model)
    using Main.trainer.LV_KAM_model.ebm_mix_prior: mix_prior, log_prior
    using Main.trainer.LV_KAM_model.MoE_likelihood: MoE_lkhood, log_likelihood
else
    error("Neither Main.LV_KAM_model nor Main.trainer.LV_KAM_model is defined")
end

include("../utils.jl")
using .Utils: device, next_rng

struct Thermodynamic_LV_KAM <: Lux.AbstractLuxLayer
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
    loss_fcn::Function
    diagnostics::Bool
end

function Lux.initialparameters(rng::AbstractRNG, model::Thermodynamic_LV_KAM)
    return (ebm = Lux.initialparameters(rng, model.prior), gen = Lux.initialparameters(rng, model.lkhood))
end

function Lux.initialstates(rng::AbstractRNG, model::Thermodynamic_LV_KAM)
    return (ebm = Lux.initialstates(rng, model.prior), gen = Lux.initialstates(rng, model.lkhood))
end

function trapz(logllhood::AbstractArray, posterior_weights::AbstractArray, Δt::AbstractArray)
    """
    Importance sampling estimators for the expected log-likelihoods,
    w.r.t the poswer posteriors at each temperature.
    """

    E_k = sum(logllhood .* posterior_weights; dims=2) # (batch_size x 1 x num_temps)
    trapz = Δt[:,:]' .* (E_k[:, 1, 1:end-1] + E_k[:, 1, 2:end]) # (batch_size x num_temps - 1)
    return 5f-1 .* sum(trapz; dims=2) 
end

function compute_mean_variance(z::AbstractArray, posterior_weights::AbstractArray)
    """
    Importance sampling estimators for mean and variance of the 
    power posterior samples at each temp. Latent dim (Q) must be on
    first dimension to make the resulting arrays contiguous. 
    """
    S, Q = size(z)
    B, _, T = size(posterior_weights)

    # Reshape for broadcasting, Q x B x T x S
    z = reshape(z, Q, 1, 1, S)
    posterior_weights = reshape(posterior_weights, 1, 1, B, T, S)

    # Importance sampling estimators for mean
    μ = sum(z .* posterior_weights[1, :, :, :, :]; dims=4)
    
    # Importance sampling estimators for variance
    diff = reshape(z .- μ, Q, B*T*S)
    vars = map(
        i -> view(diff, :, i) * view(diff, :, i)',
        1:B*T*S
    )
    Σ = reshape(reduce((x, y) -> cat(x, y; dims=3), vars), Q, Q, B, T, S)
    Σ = sum(Σ .* posterior_weights; dims=5)

    return μ[:,:,:,1], Σ[:,:,:,:,1]
end

function kl_div_2D(
    u_k::AbstractVector, 
    Σ_k::AbstractMatrix, 
    u_k1::AbstractVector, 
    Σ_k1::AbstractMatrix, 
    Q::Int; 
    ε=1f-4
    )
    """
    Compute the KL divergence between two 2D Gaussian distributions.
    LinearAlgebra is not behaving on GPU, so we need to use CPU.
    """
    eye = Matrix{Float32}(I, Q, Q) .* ε 
    Σ_k1 += eye
    Σ_k += eye

    logdet_term = logdet(Σ_k1) - logdet(Σ_k)
    trace_term = tr(Σ_k1 \ Σ_k1)
    diff = u_k1 - u_k
    quad_term = diff' * (Σ_k1 \ diff)

    return 5f-1 * (logdet_term + trace_term + quad_term - Q)
end

function compute_kl_divergence(μ::AbstractArray, Σ::AbstractArray)
    """
    Compute the KL divergences between each adjacent pair of 
    power posteriors in ascending order, assuming Gaussian.
    """
    # Adjacent pairs of power posteriors
    u_k, Σ_k = μ[:, :, 1:end-1], Σ[:, :, :, 1:end-1]
    u_k1, Σ_k1 = μ[:, :, 2:end], Σ[:, :, :, 2:end]

    Q, B, T = size(u_k)

    KL_divz = zeros(Float32, 0, T) |> device
    for b in 1:B
        KL_b = zeros(Float32, 0) |> device
        for t in 1:T
            KL_b = vcat(KL_b, kl_div_2D(
                view(u_k, :, b, t),
                view(Σ_k, :, :, b, t),
                view(u_k1, :, b, t),
                view(Σ_k1, :, :, b, t),
                Q
                ))
        end
        KL_divz = vcat(KL_divz, KL_b[:,:]')
    end
 
    return KL_divz
end

function kl_trapz(μ::AbstractArray, Σ::AbstractArray)
    """
    Compute the error term in the trapzoidal integral approximation using KL divergences.
    """

    # Ascending and descending order
    KL_divz = compute_kl_divergence(μ, Σ)
    KL_divs_reverse = compute_kl_divergence(reverse(μ, dims=3), reverse(Σ, dims=4))
    return 5f-1 .* sum(KL_divz - KL_divs_reverse; dims=2)
end

function TI_loss(
    m,
    ps, 
    st, 
    x::AbstractArray;
    seed::Int=1
    )
    """
    Compute the maximum likelihood estimation using Thermodynamic Integration.
    
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
        m.MC_samples,
        ps.ebm,
        st.ebm,
        seed
        )
        
    logprior = log_prior(m.prior, z, ps.ebm, st.ebm) # (sample_size x 1)
    logllhood = log_likelihood(m.lkhood, ps.gen, st.gen, x, z; seed=seed) # (batch_size x sample_size)
    posterior_weights = m.lkhood.weight_fcn(logllhood) # (batch_size x sample_size x num_temps)

    # Diagnostics - Thermodynamic Integration without derivatives or masking
    if m.diagnostics
        @ignore_derivatives begin

            # Trapezoidal approximation
            Δt = m.temperatures[2:end] - m.temperatures[1:end-1]
            trap_approx = trapz(logllhood, posterior_weights, Δt)

            # KL divergence term
            μ, Σ = compute_mean_variance(z, posterior_weights)
            KL_divz = kl_trapz(cpu_device()(μ), cpu_device()(Σ))
            
            println("Trapezoidal approximation: ", trap_approx)
            println("KL divergence term: ", KL_divz)
        end
    end

    # Expected power posteriors at each temperature
    loss_llhood = sum(logllhood .* posterior_weights; dims=2) 
    ex_prior = mean(logprior)
    ex_post = sum(permutedims(logprior[:,:,:], [2,1,3]) .* posterior_weights; dims=2)

    # Mask out a random temperature index for each batch, based on the variance across batch
    mask = ones(Float32, size(loss_llhood)...) |> device
    @ignore_derivatives begin
        
        # Categorically choose high variance temperature
        ranked_vars = softmax(var(loss_llhood, dims=1)[1,1,:]) |> cpu_device()
        seed = next_rng(seed)
        rand_indexes = rand(Categorical(ranked_vars), size(x, 1)) 
        
        # Mask for removal
        mask = 1 .- onehotbatch(rand_indexes, 1:length(m.temperatures)) |> device
        mask = permutedims(mask[:, :, :], [2, 3, 1])
    end

    # Tempered sum of the expected log-likelihoods 
    loss_llhood .*= mask
    loss_llhood = loss_llhood[:, 1, 2:end] - loss_llhood[:, 1, 1:end-1]
    loss_llhood = sum(loss_llhood; dims=2)

    # Tempered sum of the expected log-priors    
    ex_post .*= mask
    loss_prior = ex_post .- ex_prior
    loss_prior = loss_prior[:, 1, 2:end] - loss_prior[:, 1, 1:end-1]
    loss_prior = sum(loss_prior; dims=2)

    m.verbose && println("Prior loss: ", -mean(loss_prior), ", LLhood loss: ", -mean(loss_llhood))

    # Return the negative marginal likelihood averaged over the batch
    return -mean(loss_prior + loss_llhood)
end

end