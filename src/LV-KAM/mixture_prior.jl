module ebm_mix_prior

export mix_prior, init_mix_prior, sample_prior, log_prior

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Distributions, Accessors, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays
using NNlib: softmax, sigmoid_fast
using ChainRules: @ignore_derivatives

include("prior_sampling.jl")
include("univariate_functions.jl")
include("../utils.jl")
using .prior_sampler
using .univariate_functions
using .Utils: device, next_rng

prior_distributions = Dict(
    "uniform" => Uniform(0f0,1f0),
    "normal" => Normal(0f0,1f0),
    "bernoulli" => Bernoulli(5f-1)
)

prior_pdf = Dict(
    "uniform" => z -> 0 .<= z .<= 1,
    "normal" => z -> 1 ./ sqrt(2π) .* exp.(-z.^2 ./ 2),
    "bernoulli" => z -> 1 ./ 2
)

struct mix_prior <: Lux.AbstractLuxLayer
    fcns_qp::NamedTuple
    depth::Int
    π_0::Union{Uniform, Normal, Bernoulli}
    π_pdf::Function
    τ::Float32
    sample_z::Function
    categorical_mask::Function
    max_fcn::Function
    acceptance_fcn::Function
end

function init_mix_prior(
    conf::ConfParse;
    prior_seed::Int=1,
)
    widths = parse.(Int, retrieve(conf, "MIX_PRIOR", "layer_widths"))
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
    
    need_derivative = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps")) > 1
    τ = parse(Float32, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "gumbel_temperature"))
    ζ = parse(Float32, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "rejection_smoothening"))

    if need_derivative
        sample_function = (m, n, p, s, seed) -> sample_prior(m, n, p, s; init_seed=seed)
        choose_category = select_category_differentiable
        max_fcn = logsumexp
    else
        sample_function = (m, n, p, s, seed) -> @ignore_derivatives sample_prior(m, n, p, s; init_seed=seed)
        choose_category = select_category
        max_fcn = maximum
    end
    
    acceptance_fcn = (need_derivative ?
        (u_th, fz, f_grid) -> sigmoid_fast((u_th .- exp.(fz .- f_grid)) .* ζ) : 
        (u_th, fz, f_grid) -> u_th .< exp.(fz .- f_grid)
        )

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

    return mix_prior(functions, length(widths)-1, prior_distributions[prior_type], prior_pdf[prior_type], τ, sample_function, choose_category, max_fcn, acceptance_fcn)
end

function Lux.initialparameters(rng::AbstractRNG, prior::mix_prior)
    ps = NamedTuple(Symbol("$i") => Lux.initialparameters(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)
    @reset ps[Symbol("α")] = glorot_normal(rng, Float32, prior.fcns_qp[Symbol("1")].in_dim)
    return ps
end
 
function Lux.initialstates(rng::AbstractRNG, prior::mix_prior)
    st = NamedTuple(Symbol("$i") => Lux.initialstates(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)
    return st
end

function flip_states(
    fcn::univariate_function, 
    ps, 
    st
    )
    """
    Flip the params and states of the mixture ebm-prior.
    
    This is needed for the log-probability calculation, 
    since z_q is sampled component-wise, but needs to be
    evaluated for each component, f_{q,p}(z_q). This only works
    given that the domain of f is fixed to [0,1], (no grid updating).

    Args:
        prior: The mixture ebm-prior.
        ps: The parameters of the mixture ebm-prior.
        st: The states of the mixture ebm-prior.

    Returns:
        prior_flipped: The ebm-prior with a flipped grid.
        ps_flipped: The flipped parameters of the mixture ebm-prior.
        st_flipped: The flipped states of the mixture ebm-prior.
    """
    ps_flipped = fcn.η_trainable ? (
        coef = permutedims(ps.coef, [2, 1, 3]),
        w_base = ps.w_base',
        w_sp = ps.w_sp',
        basis_η = ps.basis_η
    ) : (
        coef = permutedims(ps.coef, [2, 1, 3]),
        w_base = ps.w_base',
        w_sp = ps.w_sp'
    )

    st_flipped = fcn.η_trainable ? st' : (
        mask = st.mask',
        basis_η = st.basis_η
    )

    grid = fcn.grid[1:1, :] # Grid is repeated along first dim for each in_dim
    @reset fcn.grid = repeat(grid, fcn.out_dim, 1)
    
    return fcn, ps_flipped, st_flipped
end

function log_prior(
    mix::mix_prior, 
    z::AbstractArray, 
    ps, 
    st
    )
    """
    Compute the unnormalized log-probability of the mixture ebm-prior.
    The likelihood of each sample, z_q, is evaluated for each component of the
    mixture model
    
    Args:
        mix: The mixture ebm-prior.
        z: The component-wise latent samples to evaulate the measure on, (num_samples, q)
        ps: The parameters of the mixture ebm-prior.
        st: The states of the mixture ebm-prior.

    Returns:
        The unnormalized log-probability of the mixture ebm-prior.
    """

    π_0 = mix.π_pdf(z)
    alpha = softmax(ps[Symbol("α")])

    for i in reverse(1:mix.depth)
        z = fwd(flip_states(mix.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")])..., z)
        z = i == mix.depth ? reshape(z, :, size(z, 3)) : sum(z, dims=2)[:, 1, :]
    end
    z = reshape(z, size(π_0)[1:2]..., :)

    # ∑_q [ log ( ∑_p α_p exp(f_{q,p}(z) ) π_0(z) ) ]
    z = exp.(z)
    prior = @tullio p[b, o, i] := alpha[i] * z[b, o, i] * π_0[b, o]
    prior = log.(sum(prior; dims=3) .+ eps(eltype(prior)))[:,:,1]
    return sum(prior; dims=2)
end

end