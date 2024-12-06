module ebm_mix_prior

export mix_prior, init_mix_prior, sample_prior, log_prior, expected_prior

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Distributions, Accessors, LuxCUDA, Statistics
using NNlib: softmax
using Flux: onehotbatch
using ChainRules: @ignore_derivatives

include("univariate_functions.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng

struct mix_prior <: Lux.AbstractLuxLayer
    fcn_qp::univariate_function
    π_tol::Float32
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
    η_trainable = spline_function == "B-spline" ? false : η_trainable
    π_tol = parse(Float32, retrieve(conf, "MIX_PRIOR", "π_tol"))

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

    return mix_prior(func, π_tol)
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

function sample_prior(prior, num_samples, ps, st; init_seed=1)
    """
    Component-wise rejection sampling for the mixture ebm-prior.

    Args:
        prior: The mixture ebm-prior.
        ps: The parameters of the mixture ebm-prior.
        st: The states of the mixture ebm-prior.

    Returns:
        z: The samples from the mixture ebm-prior, (num_samples, q). 
        seed: The updated seed.
    """

    seed = init_seed
    previous_samples = zeros(Float32, num_samples) |> device
    sample_mask = zeros(Float32, num_samples) |> device
    grid = prior.fcn_qp.grid'
    alpha = cpu_device()(softmax(ps.α))

    # Categorical component selection (per sample, per outer sum dimension)
    seed = next_rng(seed)
    rand_vals = rand(Categorical(alpha), prior.fcn_qp.out_dim, num_samples) 
    chosen_components = permutedims(collect(Float32, onehotbatch(rand_vals, 1:prior.fcn_qp.in_dim)), [3, 1, 2]) |> device # mask of 1s/0s - only admits chosen components

    # Rejection sampling
    while any(sample_mask .< 1)

        # Draw candidate samples from uniform proposal, then filter f_{q,p}(z) by chosen components
        seed = next_rng(seed) 
        z_p = rand(Uniform(0,1), num_samples, prior.fcn_qp.in_dim) |> device # z ~ Q(z)
        fz_qp = fwd(prior.fcn_qp, ps, st, z_p)
        selected_components = sum(fz_qp .* chosen_components, dims=2)[:,1,:] # samples x q

        # Grid search for max_z[ f_{q,c}(z) ]
        f_grid = @tullio fg[b, g, i, o] := fwd(prior.fcn_qp, ps, st, grid)[g ,i, o]  * chosen_components[b, i, o]
        max_f_grid = maximum(sum(f_grid; dims=3); dims=2)[:,1,1,:] # samples x q

        # Accept or reject
        seed = next_rng(seed)
        u_threshold = rand(Uniform(0,1), num_samples, prior.fcn_qp.out_dim) |> device # u ~ U(0,1)
        accept_mask = u_threshold .< exp.(selected_components .- max_f_grid) 

        # Update samples
        z_p = @tullio chosen[b, o] := z_p[b, i] * chosen_components[b, i, o]
        previous_samples = z_p .* accept_mask .* (1 .- sample_mask) .+ previous_samples .* sample_mask
        sample_mask = accept_mask .+ sample_mask
        clamp!(sample_mask, 0, 1)
    end

    return previous_samples, seed
end

function flip_states(prior::mix_prior, ps, st)
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
    ps_flipped = prior.fcn_qp.η_trainable ? (
        coef = permutedims(ps.coef, [2, 1, 3]),
        w_base = ps.w_base',
        w_sp = ps.w_sp',
        basis_η = ps.basis_η
    ) : (
        coef = permutedims(ps.coef, [2, 1, 3]),
        w_base = ps.w_base',
        w_sp = ps.w_sp'
    )

    st_flipped = prior.fcn_qp.η_trainable ? st' : (
        mask = st.mask',
        basis_η = st.basis_η
    )

    grid = prior.fcn_qp.grid[1:1, :] # Grid is repeated along first dim for each in_dim
    @reset prior.fcn_qp.grid = repeat(grid, prior.fcn_qp.out_dim, 1)
    
    return prior.fcn_qp, ps_flipped, st_flipped
end

function log_prior(mix::mix_prior, z, ps, st)
    """
    Compute the unnormalized log-probability of the mixture ebm-prior.
    Log-sum-exp trick is used for numerical stability.
    
    Args:
        mix: The mixture ebm-prior.
        z: The component-wise latent samples to evaulate the measure on, (num_samples, q)
        ps: The parameters of the mixture ebm-prior.
        st: The states of the mixture ebm-prior.

    Returns:
        The unnormalized log-probability of the mixture ebm-prior.
    """

    π_0 = 0 .<= z .<= 1 # π_0(z) = U(z;0,1)
    π_0 = Float32.(π_0) 
    alpha = softmax(ps.α)
    f_qp = fwd(flip_states(mix, ps, st)..., z)
    
    # ∑_q [ log ( ∑_p α_p exp(f_{q,p}(z) ) π_0(z) ) ]
    # max_f = maximum(f_qp; dims=3)
    # exp_shifted = exp.(f_qp .- max_f)
    # prior = @tullio p[b, o, i] := alpha[i] * exp_shifted[b, o, i] * π_0[b, o]
    # prior = sum(prior; dims=3)
    # log_prior = max_f .+ log.(prior .+ mix.π_tol)

    exp_f = exp.(f_qp)
    prior = @tullio p[b, o, i] := alpha[i] * exp_f[b, o, i] * π_0[b, o]
    log_prior = log.(sum(prior; dims=3) .+ mix.π_tol)
    
    return sum(log_prior; dims=2)[:,1,1]
end

function expected_prior(prior::mix_prior, num_samples, ps, st, ρ_fcn; seed=1)
    """
    Compute the expected prior of an arbritrary function of the latent variable, 
    using a Monte Carlo estimator. Sampling procedure is ignored from the gradient computation.
    """
    z, seed = @ignore_derivatives sample_prior(prior, num_samples, ps, st; init_seed=seed)
    return mean(ρ_fcn(z, ps)), seed
end

end