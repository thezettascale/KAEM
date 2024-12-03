module ebm_mix_prior

export mix_prior, init_mix_prior, sample_prior

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Zygote, Distributions, Accessors, LuxCUDA
using NNlib: softmax
using Flux: onehotbatch

include("univariate_functions.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng

struct mix_prior <: Lux.AbstractLuxLayer
    fcn_qp::univariate_function
end

function init_mix_prior(
    conf::ConfParse;
    prior_seed::Int=1,
)
    p = parse(Int, retrieve(conf, "MIX_PRIOR", "latent_dim"))
    q = parse(Int, retrieve(conf, "MIX_PRIOR", "hidden_dim"))
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

    prior_seed = next_rng(prior_seed)
    base_scale = (μ_scale * (1f0 / √(Float32(p))) 
    .+ σ_base .* (randn(Float32, p, q) .* 2f0 .- 1f0) .* (1f0 / √(Float32(p))))

    func = init_function(
        p,
        q;
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

    return mix_prior(func)
end

function Lux.initialparameters(rng::AbstractRNG, prior::mix_prior)
    ps =  Lux.initialparameters(rng, prior.fcn_qp)
    @reset ps[:α] = glorot_normal(rng, Float32, prior.fcn_qp.in_dim)
    return ps
end
 
function Lux.initialstates(rng::AbstractRNG, prior::mix_prior)
    st = Lux.initialstates(rng, prior.fcn_qp)
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

    seed = init_seed
    previous_samples = zeros(Float32, num_samples) |> device
    sample_mask = zeros(Float32, num_samples) |> device
    grid = prior.fcn_qp.grid'
    alpha = cpu_device()(softmax(ps.α))

    while any(sample_mask .< 1)
        # Categorical component selection (per sample, per outer sum dimension)
        seed = next_rng(seed)
        rand_vals = rand(Categorical(alpha), prior.fcn_qp.out_dim, num_samples) 
        chosen_components = permutedims(collect(Float32, onehotbatch(rand_vals, 1:prior.fcn_qp.in_dim)), [3, 1, 2]) |> device # mask of 1s/0s - only admits chosen components

        # Draw candidate samples from Gaussian proposal, filter f_{q,p}(z) + z^2/2 by chosen components
        seed = next_rng(seed) 
        z_p = rand(Normal(0,1), num_samples, prior.fcn_qp.in_dim) |> device # z ~ Q(z)
        fz_qp = @tullio f[b, i, o] := fwd(prior.fcn_qp, ps, st, z_p)[b, i, o] + ((z_p[b, i]^2)/2) 
        selected_components = sum(fz_qp .* chosen_components, dims=2)[:,1,:] # samples x q

        # Grid search for max_z[ f_{q,c}(z) + z^2/2 ]
        f_grid = @tullio fg[b, i, o] := fwd(prior.fcn_qp, ps, st, grid)[b ,i, o] + ((grid[b, i]^2)/2)
        f_grid = @tullio fg[b, g, o] := f_grid[g, i, o] * chosen_components[b, i, o]
        max_f_grid = maximum(f_grid; dims=2)[:,1,:] # samples x q

        # Accept or reject
        seed = next_rng(seed)
        u_threshold = rand(Uniform(0,1), num_samples, prior.fcn_qp.out_dim) |> device # u ~ U(0,1)
        accept_mask = u_threshold .< exp.(selected_components .- max_f_grid) 

        # Update samples
        z_p = @tullio chosen[b, o] := z_p[b, i] * chosen_components[b, i, o]
        previous_samples = z_p .* accept_mask .* (1 .- sample_mask) .+ previous_samples .* sample_mask
        sample_mask = accept_mask .+ sample_mask
    end

    return previous_samples, seed
end

end