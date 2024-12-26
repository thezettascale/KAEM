module ebm_mix_prior

export mix_prior, init_mix_prior, sample_prior, log_prior

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Distributions, Accessors, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays
using NNlib: softmax, sigmoid_fast
using Flux: onehotbatch
using ChainRules: @ignore_derivatives

include("univariate_functions.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng

prior_distributions = Dict(
    "uniform" => Uniform(0f0,1f0),
    "normal" => Normal(0f0,1f0),
    "bernoulli" => Bernoulli(5f-1)
)

prior_pdf = Dict(
    "uniform" => z -> 0 .<= z .<= 1 .|> Float32,
    "normal" => z -> 1 ./ sqrt(2π) .* exp.(-z.^2 ./ 2),
    "bernoulli" => z -> 1 ./ 2
)

struct mix_prior <: Lux.AbstractLuxLayer
    fcns_qp::NamedTuple
    depth::Int
    π_0::Union{Uniform, Normal, Bernoulli}
    π_pdf::Function
    sample_z::Function
end

function choose_component(alpha, num_samples, q_size, p_size; seed=1)
    """
    Creates a one-hot mask for each mixture model, q, to select one component, p.
    
    Args:
        alpha: The mixture proportions, (q, p).
        num_samples: The number of samples to generate.
        q_size: The number of mixture models.
        seed: The seed for the random number generator.

    Returns:
        chosen_components: The one-hot mask for each mixture model, (num_samples, q, p).
        seed: The updated seed.
    """
    seed, rng = next_rng(seed)
    rand_vals = rand(rng, Uniform(0,1), q_size, num_samples) 
    
    function categorical_mask(α, rv)
        """Returns sampled indices from a categorical distribution on alpha."""
        idxs = map(u -> findfirst(x -> x >= u, α), rv)
        idxs = reduce(vcat, idxs)
        idxs = ifelse.(isnothing.(idxs), p_size, idxs)
        idxs = collect(Float32, onehotbatch(idxs, 1:p_size))   
        return permutedims(idxs[:,:,:], [2, 3, 1])
    end
    
    chosen_components = map(i -> categorical_mask(view(alpha, i, :), view(rand_vals, i, :)), 1:q_size)
    return reduce(hcat, chosen_components) |> device, seed
end

function sample_prior(
    prior::mix_prior,
    num_samples::Int, 
    ps,
    st;
    seed::Int=1
    )
    """
    Component-wise inverse transform sampling for the mixture ebm-prior.
    p = components of mixture model
    q = number of mixture models

    Args:
        prior: The mixture ebm-prior.
        ps: The parameters of the mixture ebm-prior.
        st: The states of the mixture ebm-prior.

    Returns:
        z: The samples from the mixture ebm-prior, (num_samples, q). 
        seed: The updated seed.
    """
    p_size = prior.fcns_qp[Symbol("$(prior.depth)")].out_dim
    q_size = prior.fcns_qp[Symbol("1")].in_dim
    
    # Categorical component selection (per sample, per outer sum dimension)
    component_mask, seed = choose_component(
        cpu_device()(cumsum(softmax(ps[Symbol("α")]; dims=2); dims=2)),
        num_samples,
        q_size,
        p_size;
        seed=seed
    )

    # Evaluate prior on grid [0,1]
    grid = prior.fcns_qp[Symbol("1")].grid'
    f_grid = grid
    Δg = f_grid[2:end, :] .- f_grid[1:end-1, :] 
    π_grid = prior.π_pdf(f_grid)
    grid_size = size(f_grid, 1)
    for i in 1:prior.depth
        f_grid = fwd(prior.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], f_grid)
        f_grid = i == 1 ? reshape(f_grid, grid_size*q_size, size(f_grid, 3)) : sum(f_grid, dims=2)[:, 1, :] 
    end
    f_grid = reshape(f_grid, grid_size, q_size, p_size)
    @tullio exp_fg[b, g, q] := exp(f_grid[g, q, p]) * π_grid[g, q] * component_mask[b, q, p]

    # CDF evaluated by trapezium rule for integration; 1/2 * (u(z_{i+1}) + u(z_i)) * Δx
    trapz = 5f-1 .* permutedims(Δg[:,:,:], [3,1,2]) .* (exp_fg[:, 2:end, :] .+ exp_fg[:, 1:end-1, :]) 
    cdf = cumsum(trapz, dims=2) ./ sum(trapz, dims=2)

    # Inverse transform sampling
    seed, rng = next_rng(seed)
    rand_vals = rand(rng, Uniform(0,1), num_samples, q_size) |> device
    @tullio geq_indices[b, g, q] := cdf[b, g, q] >= rand_vals[b, q]

    function sample_mixture(q)
        """Returns samples from a given mixture model, q."""
        # Index of trapz where CDF >= rand_val
        idxs = map(i -> findfirst(view(geq_indices, i, :, q)), 1:num_samples)
        idxs = reduce(vcat, idxs)
        idxs = ifelse.(isnothing.(idxs), grid_size-1, idxs)

        # Interpolate between given trapezium
        cdf_q, u_q = view(cdf, :, :, q), view(rand_vals, :, q)
        mask1 = collect(Float32, onehotbatch(idxs, 1:grid_size-1)) |> device
        mask2 = collect(Float32, onehotbatch(idxs, 2:grid_size)) |> device
        @tullio cd1[b] := cdf_q[b,g] * mask1[g,b]
        @tullio cd2[b] := cdf_q[b,g] * mask2[g,b]

        # Linear interpolation
        z1, z2 = grid[idxs, q], grid[idxs .+ 1, q]
        return (z1 .+ (z2 .- z1) .* ((u_q .- cd1) ./ (cd2 .- cd1)))
    end

    return reduce(hcat, map(i -> sample_mixture(i)[:,:], 1:q_size)), seed
end

function log_prior(
    mix::mix_prior, 
    z::AbstractArray, 
    ps, 
    st
    )
    """
    Compute the unnormalized log-probability of the mixture ebm-prior.
    The likelihood of samples from each mixture model, z_q, is evaluated 
    for all components of the mixture model it has been sampled from , M_q.
    
    Args:
        mix: The mixture ebm-prior.
        z: The component-wise latent samples to evaulate the measure on, (num_samples, q)
        ps: The parameters of the mixture ebm-prior.
        st: The states of the mixture ebm-prior.

    Returns:
        The unnormalized log-probability of the mixture ebm-prior.
    """
    b_size, q_size, p_size = size(z)..., mix.fcns_qp[Symbol("$(mix.depth)")].out_dim
    alpha = softmax(ps[Symbol("α")]; dims=2) # Mixture proportions and prior
    π_0 = mix.π_pdf(z) 

    # Energy functions of each component, q -> p
    for i in 1:mix.depth
        z = fwd(mix.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = i == 1 ? reshape(z, b_size*q_size, size(z, 3)) : sum(z, dims=2)[:, 1, :]
    end
    z = reshape(z, b_size, q_size, p_size)

    # ∑_q [ log ( ∑_p α_p exp(f_{q,p}(z_q)) π_0(z_q) ) ] ; likelihood of samples under each component
    @tullio prob[b, q] := alpha[q, p] * exp(z[b, q, p]) * π_0[b, q]
    return sum(log.(prob .+ eps(eltype(prob))); dims=2)[:,1] # Sum over independent log-mixture models
end

function init_mix_prior(
    conf::ConfParse;
    prior_seed::Int=1,
)
    widths = parse.(Int, retrieve(conf, "MIX_PRIOR", "layer_widths"))
    widths = reverse(widths)
    spline_degree = parse(Int, retrieve(conf, "MIX_PRIOR", "spline_degree"))
    base_activation = retrieve(conf, "MIX_PRIOR", "base_activation")
    spline_function = retrieve(conf, "MIX_PRIOR", "spline_function")
    grid_size = parse(Int, retrieve(conf, "MIX_PRIOR", "grid_size"))
    grid_update_ratio = parse(Float32, retrieve(conf, "MIX_PRIOR", "grid_update_ratio"))
    grid_range = parse.(Float32, retrieve(conf, "MIX_PRIOR", "grid_range"))
    ε_scale = parse(Float32, retrieve(conf, "MIX_PRIOR", "ε_scale"))
    μ_scale = parse(Float32, retrieve(conf, "MIX_PRIOR", "μ_scale"))
    σ_base = parse(Float32, retrieve(conf, "MIX_PRIOR", "σ_base"))
    σ_spline = parse(Float32, retrieve(conf, "MIX_PRIOR", "σ_spline"))
    init_η = parse(Float32, retrieve(conf, "MIX_PRIOR", "init_η"))
    η_trainable = parse(Bool, retrieve(conf, "MIX_PRIOR", "η_trainable"))
    η_trainable = spline_function == "B-spline" ? false : η_trainable
    prior_type = retrieve(conf, "MIX_PRIOR", "π_0")
    
    sample_function = (m, n, p, s, seed) -> @ignore_derivatives sample_prior(m, n, p, s; seed=seed)
    
    functions = NamedTuple()
    for i in eachindex(widths[1:end-1])
        prior_seed, rng = next_rng(prior_seed)
        base_scale = (μ_scale * (1f0 / √(Float32(widths[i])))
        .+ σ_base .* (randn(rng, Float32, widths[i], widths[i+1]) .* 2f0 .- 1f0) .* (1f0 / √(Float32(widths[i]))))

        func = init_function(
        widths[i],
        widths[i+1];
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

        @reset functions[Symbol("$i")] = func
    end

    return mix_prior(functions, length(widths)-1, prior_distributions[prior_type], prior_pdf[prior_type], sample_function)
end

function Lux.initialparameters(rng::AbstractRNG, prior::mix_prior)
    q_size = prior.fcns_qp[Symbol("1")].in_dim
    p_size = prior.fcns_qp[Symbol("$(prior.depth)")].out_dim
    ps = NamedTuple(Symbol("$i") => Lux.initialparameters(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)
    @reset ps[Symbol("α")] = glorot_uniform(Float32, q_size, p_size)
    return ps
end
 
function Lux.initialstates(rng::AbstractRNG, prior::mix_prior)
    st = NamedTuple(Symbol("$i") => Lux.initialstates(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)
    return st
end

end