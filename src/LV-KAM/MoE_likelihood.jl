module MoE_likelihood

export MoE_lkhood, init_MoE_lkhood, log_likelihood, expected_posterior, generate_from_z 

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Distributions, Accessors, LuxCUDA, Statistics, LinearAlgebra
using NNlib: softmax, sigmoid_fast, tanh_fast
using ChainRules: @ignore_derivatives

include("univariate_functions.jl")
include("mixture_prior.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng
using .ebm_mix_prior

activation_mapping = Dict(
    "tanh" => tanh_fast,
    "sigmoid" => sigmoid_fast,
    "none" => x -> x 
)

lkhood_models = Dict(
    "l2" => (x, x̂) -> -sum((x̂ .- x).^2, dims=2),
)

struct MoE_lkhood <: Lux.AbstractLuxLayer
    fcn_q::univariate_function
    out_size::Int
    σ_ε::Float32
    σ_llhood::Float32
    log_lkhood_model::Function
    output_activation::Function
    weight_fcn::Function
end

function init_MoE_lkhood(
    conf::ConfParse,
    output_dim::Int;
    lkhood_seed::Int=1,
)
    q = parse(Int, retrieve(conf, "MIX_PRIOR", "hidden_dim"))
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
    output_act = retrieve(conf, "MOE_LIKELIHOOD", "output_activation")

    need_derivative = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps")) > 1
    weight_fcn = need_derivative ? x -> softmax(x; dims=3) :  x -> @ignore_derivatives softmax(x; dims=3)

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
    
    return MoE_lkhood(func, output_dim, noise_var, gen_var, lkhood_models[lkhood_model], activation_mapping[output_act], weight_fcn)
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

function generate_from_z(lkhood::MoE_lkhood, ps, st, z; seed=1)
    """
    Generate data from the likelihood model.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        seed: The seed for the random number generator.

    Returns:
        The generated data.
        The updated seed.
    """
    # Gen function, experts for all features
    Λ = fwd(lkhood.fcn_q, ps, st, z)
    seed = next_rng(seed)
    ε = rand(Normal(0f0, lkhood.σ_ε), size(lkhood.out_size)) |> device

    # Gating function, feature-specific
    wz = @tullio out[b, i, o] := z[b, i] * ps.gate_w[i, o]
    γ = softmax(wz; dims=2)

    # Generate data
    x̂ = @tullio gen[b, i, o] := Λ[b, i, 1] * γ[b, i, o]
    x̂ = sum(x̂, dims=2)[:, 1, :] .+ ε
    return lkhood.output_activation(x̂), seed
end

function log_likelihood(lkhood::MoE_lkhood, ps, st, x, z; seed=1)
    """
    Compute the log-likelihood of the data given the latent variable.
    The updated seed is not returned, since noise is ignored by derivatives anyway.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data, (batch_size, out_dim).
        z: The latent variable, (batch_size*num_latent_samples, q).
        seed: The seed for the random number generator.

    Returns:
        The log-likelihood per sample of data given the latent variable.
    """
    
    x̂, seed = generate_from_z(lkhood, ps, st, z; seed=seed)
    x̂ = reshape(x̂, size(x)..., fld(size(x̂, 1), size(x, 1))) # (batch_size, out_dim, num_latent_samples)    
    return lkhood.log_lkhood_model((x, size(x)..., 1), x̂) ./ (2*lkhood.σ_llhood^2)
end

function expected_posterior(prior, lkhood, ps, st, x, ρ_fcn, ρ_ps; seed=1, t=device([1f0]))
    """
    Compute the expected posterior of an arbritrary function of the latent variable,
    using importance sampling. Sampling procedure is ignored from the gradient computation.

    Args:
        prior: The prior distribution of the latent variable.
        lkhood: The likelihood model.
        ps: The parameters of the LV-KAM.
        st: The states of the LV-KAM.
        x: The data.
        ρ_fcn: The function of the latent variable to compute the expected posterior. Should return a sample_size x 1 array.
        seed: The seed for the random number generator.

    Returns:
        The expected posterior of the function of the latent variable, (w.r.t samples from the latent space, NOT BATCH).
        The updated seed. 
    """

    # MC estimator is mapped over batch dim for memory efficiency
    num_iters = fld(size(x, 1), prior.MC_batch_size)
    function MC_estimate(x_i)
        z, seed = prior.sample_z(prior, prior.num_latent_samples*prior.MC_batch_size, ps.ebm, st.ebm, seed)
        weights = lkhood.weight_fcn(view(t, 1, length(t), 1) .* log_likelihood(lkhood, ps.gen, st.gen, x_i, z; seed=seed))
        return sum(ρ_fcn(z, x_i, ρ_ps) .* weights; dims=3)
    end
    
    ρ = map(i -> MC_estimate(view(x, i:i+prior.MC_batch_size-1, :)), 1:num_iters)
    return vcat(ρ...), seed
end

end