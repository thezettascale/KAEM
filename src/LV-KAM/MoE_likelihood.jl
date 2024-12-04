module MoE_likelihood

export MoE_lkhood, init_MoE_lkhood, log_likelihood, expected_posterior  

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Distributions, Accessors, LuxCUDA, Statistics
using NNlib: softmax
using ChainRules: @ignore_derivatives

include("univariate_functions.jl")
include("mixture_prior.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng
using .ebm_mix_prior

lkhood_models = Dict(
    "l2" => (x, x̂) -> sum((x .- x̂) .^ 2, dims=2)[:, 1],
    "bernoulli" => (x, x̂) -> sum(x .* log.(x̂) .+ (1 .- x) .* log.(1 .- x̂), dims=2)[:, 1],
)

struct MoE_lkhood <: Lux.AbstractLuxLayer
    fcn_q::univariate_function
    out_size::Int
    σ_ε::Float32
    σ_llhood::Float32
    log_lkhood_model::Function
end

function init_MoE_lkhood(
    conf::ConfParse;
    lkhood_seed::Int=1,
)
    q = parse(Int, retrieve(conf, "MIX_PRIOR", "hidden_dim"))
    o = parse(Int, retrieve(conf, "MOE_LIKELIHOOD", "output_dim"))
    spline_degree = parse(Int, retrieve(conf, "MOE_LIKELIHOOD", "spline_degree"))
    base_activation = retrieve(conf, "MOE_LIKELIHOOD", "base_activation")
    spline_function = retrieve(conf, "MOE_LIKELIHOOD", "spline_function")
    grid_size = parse(Int, retrieve(conf, "MOE_LIKELIHOOD", "grid_size"))
    grid_update_ratio = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "grid_update_ratio"))
    grid_range = parse.(Float32, retrieve(conf, "MOE_LIKELIHOOD", "grid_range"))
    ε_scale = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "ε_scale"))
    μ_scale = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "μ_scale"))
    σ_base = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "σ_base"))
    σ_spline = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "σ_spline"))
    init_η = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "init_η"))
    η_trainable = parse(Bool, retrieve(conf, "MOE_LIKELIHOOD", "η_trainable"))
    η_trainable = spline_function == "B-spline" ? false : η_trainable
    noise_var = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "generator_noise_variance"))
    gen_var = parse(Float32, retrieve(conf, "MOE_LIKELIHOOD", "generator_variance"))
    lkhood_model = retrieve(conf, "MOE_LIKELIHOOD", "likelihood_model")

    lkhood_seed = next_rng(lkhood_seed)
    base_scale = (μ_scale * (1f0 / √(Float32(q)))
    .+ σ_base .* (randn(Float32, q, 1) .* 2f0 .- 1f0) .* (1f0 / √(Float32(q))))

    func = init_function(
        q,
        1;
        spline_degree=spline_degree,
        base_activation=base_activation,
        spline_function=spline_function,
        grid_size=grid_size,
        grid_update_ratio=grid_update_ratio,
        grid_range=Tuple(grid_range),
        ε_scale=ε_scale,
        σ_base=base_scale,
        σ_spline=σ_spline,
        init_η=init_η,
        η_trainable=η_trainable,
    )
    
    return MoE_lkhood(func, o, noise_var, gen_var, lkhood_models[lkhood_model])
end

function Lux.initialparameters(rng::AbstractRNG, lkhood::MoE_lkhood)
    ps = Lux.initialparameters(rng, lkhood.fcn_q)
    @reset ps[:gate_w] = glorot_normal(rng, Float32, lkhood.fcn_q.in_dim, lkhood.out_size)
    return ps
end

function Lux.initialstates(rng::AbstractRNG, lkhood::MoE_lkhood)
    st = Lux.initialstates(rng, lkhood.fcn_q)
    return st
end

function log_likelihood(lkhood::MoE_lkhood, ps, st, x, z; seed=1)
    """
    Compute the log-likelihood of the data given the latent variable.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        seed: The seed for the random number generator.

    Returns:
        The log-likelihood per sample of data given the latent variable.
    """
    
    # Gen function, experts for all features
    Λ = fwd(lkhood.fcn_q, ps, st, z)
    seed = next_rng(seed)
    ε = rand(Normal(0f0, lkhood.σ_ε), size(x)) |> device

    # Gating function, feature-specific
    wz = @tullio out[b, i, o] := z[b, i] * ps.gate_w[i, o]
    γ = softmax(wz; dims=2)

    # Generate data, and log-likelihood
    x̂ = @tullio gen[b, i, o] := Λ[b, i, 1] * γ[b, i, o]
    x̂ = sum(x̂, dims=2)[:, 1, :] .+ ε
    return lkhood.log_lkhood_model(x̂, x) ./ lkhood.σ_llhood
end

function expected_posterior(prior, lkhood, ps, st, x, ρ_fcn, ρ_ps; seed=1)
    """
    Compute the expected posterior of an arbritrary function of the latent variable,
    using importance sampling. Sampling procedure is ignored from the gradient computation.

    Args:
        prior: The prior distribution of the latent variable.
        lkhood: The likelihood model.
        ps: The parameters of the LV-KAM.
        st: The states of the LV-KAM.
        x: The data.
        ρ_fcn: The function of the latent variable to compute the expected posterior.
        seed: The seed for the random number generator.

    Returns:
        The expected posterior of the function of the latent variable.  
        The updated seed. 
    """

    prior_ps, prior_st = ps.ebm, st.ebm
    gen_ps, gen_st = ps.gen, st.gen

    z, seed = @ignore_derivatives sample_prior(prior, size(x,1), prior_ps, prior_st; init_seed=seed)
    ρ_values = ρ_fcn(z, ρ_ps)
    weights = @ignore_derivatives softmax(log_likelihood(lkhood, gen_ps, gen_st, x, z; seed=seed))

    return sum(ρ_values .* weights), seed
end

end