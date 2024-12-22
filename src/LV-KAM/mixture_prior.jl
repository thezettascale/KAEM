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

function sample_prior(
    prior::mix_prior,
    num_samples::Int, 
    ps,
    st;
    seed::Int=1
    )
    """
    Component-wise rejection sampling for the mixture ebm-prior.
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

    previous_samples = zeros(Float32, num_samples) |> device
    sample_mask = zeros(Float32, num_samples) |> device
    p_size = prior.fcns_qp[Symbol("$(prior.depth)")].out_dim
    q_size = prior.fcns_qp[Symbol("1")].in_dim
    
    # Categorical component selection (per sample, per outer sum dimension)
    alpha = cpu_device()(cumsum(softmax(ps[Symbol("α")]; dims=2); dims=2))
    seed = next_rng(seed)
    rand_vals = rand(Uniform(0,1), q_size, num_samples) 
    
    function categorical_mask(α, rv)
        """Creates a one-hot mask for each mixture model, q, to select one component, p."""
        idxs = map(u -> findfirst(x -> x >= u, α), rv)
        idxs = reduce(vcat, idxs)
        idxs = ifelse.(isnothing.(idxs), p_size, idxs)
        idxs = collect(Float32, onehotbatch(idxs, 1:p_size))   
        return permutedims(idxs[:,:,:], [2, 3, 1])
    end
    
    chosen_components = map(i -> categorical_mask(view(alpha, i, :), view(rand_vals, i, :)), 1:q_size)
    chosen_components = reduce(hcat, chosen_components) |> device

    # Rejection sampling
    while any(sample_mask .< 5f-1)

        # Draw candidate samples from proposal, i.e. prior
        seed = next_rng(seed) 
        z = rand(prior.π_0, num_samples, q_size) |> device # z ~ Q(z)

        # Forward pass of proposal samples through mixture model; p -> q
        fz_qp = z
        for i in 1:prior.depth
            fz_qp = fwd(prior.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], fz_qp)
            fz_qp = i == 1 ? reshape(fz_qp, num_samples*q_size, size(fz_qp, 3)) : sum(fz_qp, dims=2)[:, 1, :]
        end
        fz_qp = reshape(fz_qp, num_samples, q_size, p_size)

        # Forward pass of grid [0,1] through model
        f_grid = prior.fcns_qp[Symbol("1")].grid'
        grid_size = size(f_grid, 1)
        for i in 1:prior.depth
            f_grid = fwd(prior.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], f_grid)
            f_grid = i == 1 ? reshape(f_grid, grid_size*q_size, size(f_grid, 3)) : sum(f_grid, dims=2)[:, 1, :] 
        end
        f_grid = reshape(f_grid, grid_size, q_size, p_size)

        # Filter chosen components of mixture model, (samples x q)
        fz_qp = sum(fz_qp .* chosen_components, dims=3)[:,:,1]

        # Grid search for max_z[ f_{q,c}(z) ] for chosen components
        @tullio f_g[b, g, q] := f_grid[g, q, p]  * chosen_components[b, q, p]
        f_g = maximum(f_g; dims=2)[:,1,:] # Filtered max f_qp, (samples x q)

        # Accept or reject
        seed = next_rng(seed)
        u_threshold = rand(Uniform(0,1), num_samples, q_size) |> device # u ~ U(0,1)
        accept_mask = u_threshold .< exp.(fz_qp - f_g)

        # Update samples
        previous_samples = z .* accept_mask .* (1 .- sample_mask) .+ previous_samples .* sample_mask
        sample_mask = accept_mask .* (1 .- sample_mask) .+ sample_mask
        clamp.(sample_mask, 0f0, 1f0)
    end

    return previous_samples, seed
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
    
    # Mixture proportions and prior, prepared for broadcasting
    alpha = softmax(ps[Symbol("α")]; dims=2)
    alpha = permutedims(alpha[:,:,:], [3, 1, 2])
    π_0 = mix.π_pdf(z)[:,:,:]

    # Energy functions of each component, q -> p
    for i in 1:mix.depth
        z = fwd(mix.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = i == 1 ? reshape(z, b_size*q_size, size(z, 3)) : sum(z, dims=2)[:, 1, :]
    end
    z = reshape(z, b_size, q_size, p_size)

    # ∑_q [ log ( ∑_p α_p exp(f_{q,p}(z_q)) π_0(z_q) ) ] ; likelihood of samples under each component
    z = sum(alpha .* exp.(z) .* π_0, dims=3) # Sum over components
    return sum(log.(z .+ eps(eltype(z))); dims=2)[:,1,1] # Sum over independent log-mixture models
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
        prior_seed = next_rng(prior_seed)
        base_scale = (μ_scale * (1f0 / √(Float32(widths[i])))
        .+ σ_base .* (randn(Float32, widths[i], widths[i+1]) .* 2f0 .- 1f0) .* (1f0 / √(Float32(widths[i]))))

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