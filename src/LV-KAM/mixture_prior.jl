module ebm_mix_prior

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Zygote, Distributions
using NNlib: softmax
using Flux: onehotbatch

include("univariate_functions.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device

struct mix_prior <: Lux.AbstractLuxLayer
    fcn_qp::univariate_function
end

function init_mix_prior(
    conf::ConfParse;
    prior_seed::Int=1,
)
    widths = parse.(Int, retrieve(conf, "MIX_PRIOR", "layer_widths")) 
    depth = length(widths) - 1

    spline_degree = parse(Int, retrieve(conf, "MIX_PRIOR", "spline_degree"))
    base_activation = retrieve(conf, "MIX_PRIOR", "base_activation")
    spline_function = retrieve(conf, "MIX_PRIOR", "spline_function")
    grid_size = parse(Int, retrieve(conf, "MIX_PRIOR", "grid_size"))
    grid_update_ratio = parse(Float32, retrieve(conf, "MIX_PRIOR", "grid_update_ratio"))
    grid_range = parse.(Float32, retrieve(conf, "MIX_PRIOR", "grid_range"))
    ε_scale = parse(Float32, retrieve(conf, "MIX_PRIOR", "ε_scale"))
    μ_scale = parse(Float32, retrieve(conf, "MIX_PRIOR", "μ_scale"))
    σ_base = parse(Float32, split(retrieve(conf, "MIX_PRIOR", "σ_base"), ","))
    σ_spline = parse(Float32, retrieve(conf, "MIX_PRIOR", "σ_spline"))
    init_η = parse(Float32, split(retrieve(conf, "MIX_PRIOR", "init_η"), ","))
    η_trainable = parse(Bool, retrieve(conf, "MIX_PRIOR", "η_trainable"))

    Random.seed!(prior_seed)
    base_scale = (μ_scale * (1f0 / √(Float32(widths[i]))) 
    .+ σ_scale .* (randn(Float32, widths[i], widths[i + 1]) .* 2f0 .- 1f0) .* (1f0 / √(Float32(widths[i]))))

    func = univariate_function(
        widths[i],
        widths[i + 1],
        spline_degree,
        base_activation,
        spline_function,
        rand(Float32, grid_size, widths[i]),
        grid_size,
        grid_update_ratio,
        grid_range,
        ε_scale,
        σ_base,
        σ_spline,
        init_η,
        η_trainable,
    )


    return mix_prior(func)
end

function Lux.initialparameters(rng::AbstractRNG, prior::mix_prior)
    ps = (fcn_ps => Lux.initialparameters(rng, prior.fcn_qp))
    ps = merge(ps, NamedTuple(α => glorot_normal(rng, Float32, fcn_qp.in_dim))) # Mixture proportions
    return ps
end

function Lux.initialstates(rng::AbstractRNG, prior::mix_prior)
    st = NamedTuple(fcn_st => Lux.initialstates(rng, prior.fcn_qp))
    return st
end

function sample_prior(prior::mix_prior, num_samples, ps, st; init_seed=1)
    """
    Component-wise rejection sampling for the mixture ebm-prior.

    Args:
        prior: The mixture ebm-prior.
        ps: The parameters of the mixture ebm-prior.
        st: The states of the mixture ebm-prior.

    Returns:
        z: The samples from the mixture ebm-prior, (num_samples, q). 
    """

    # Choose components per sample, per outer sum dimension (q)
    Random.seed!(init_seed) 
    chosen_components = rand(Categorical(softmax(ps.α)), prior.out_dim, num_samples) |> device # num_samples x q
    chosen_components = permutedims(onehotbatch(chosen_components, 1:prior.in_dim), [3, 1, 2]) # num_samples x p x q
    init_seed += 1

    # Draw candidate samples from Gaussian prior, and rejection variable
    z_p = rand(Float32, Normal(0,1), prior.num_samples, prior.fcn_qp.in_dim) |> device # z ~ π_0(z)
    u_threshold = rand(Float32, Uniform(0,1), prior.num_samples, prior.fcn_qp.out_dim) |> device # u ~ U(0,1)

    fz_qp = fwd(prior.fcn_qp, ps.fcn_ps, st.fcn_st, z_p) # f_{q,p}(z) # samples x p x q
    selected_components = sum(fz_qp .* chosen_components, dims=2)[:,1,:] # samples x q
end

end