module ebm_ebm_prior

export ebm_prior, init_ebm_prior, log_prior

using CUDA, KernelAbstractions, Tullio, FastGaussQuadrature
using ConfParser, Random, Distributions, Lux, Accessors, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays
using NNlib: softmax, sigmoid_fast
using ChainRules: @ignore_derivatives

include("univariate_functions.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng, half_quant, full_quant, removeZero, removeNeg, hq, fq

prior_pdf = Dict(
    "uniform" => z -> half_quant.(0 .<= z .<= 1) |> device,
    "gaussian" => z -> half_quant(1 ./ sqrt(2π)) .* exp.(-z.^2 ./ 2),
    "lognormal" => (z, ε) -> exp.(-(log.(z .+ ε)).^2 ./ 2) ./ (z .* half_quant(sqrt(2π)) .+ ε),
    "ebm" => z -> ones(half_quant, size(z)) |> device,
)

struct ebm_prior <: Lux.AbstractLuxLayer
    fcns_qp::NamedTuple
    layernorm::Bool
    depth::Int
    prior_type::AbstractString
    π_pdf::Function
    sample_z::Function
    p_size::Int
    q_size::Int
    quad::Function
    N_quad::Int
    nodes::AbstractArray{half_quant}
    weights::AbstractArray{full_quant}
end

function prior_fwd(ebm, ps, st, z::AbstractArray{half_quant})
    """Forward pass through the ebm-prior, returning the energy function"""
    for i in 1:ebm.depth
        z = fwd(ebm.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = i == 1 ? reshape(z, size(z, 2), ebm.p_size*size(z, 3)) : dropdims(sum(z, dims=1); dims=1)

        if ebm.layernorm && i < ebm.depth
            z, st_new = Lux.apply(ebm.fcns_qp[Symbol("ln_$i")], z, ps[Symbol("ln_$i")], st[Symbol("ln_$i")])
            @reset st[Symbol("ln_$i")] = st_new
        end
    end
    return reshape(z, ebm.q_size, ebm.p_size, :), st
end

function trapezium_quadrature(ebm, ps, st; ε::half_quant=eps(half_quant))
    """Trapezoidal rule for numerical integration"""

    # Evaluate prior on grid [0,1]
    f_grid = st[Symbol("1")].grid
    grid = f_grid |> cpu_device() .|> full_quant
    Δg = f_grid[:, 2:end] - f_grid[:, 1:end-1] .|> full_quant
    
    π_grid = ebm.prior_type == "lognormal" ? ebm.π_pdf(f_grid, ε) : ebm.π_pdf(f_grid)
    grid_size = size(f_grid, 2)

    # Energy function of each component, q -> p
    f_grid, st = prior_fwd(ebm, ps, st, f_grid)

    # Filter out components
    @tullio exp_fg[q, p, g] := (exp(f_grid[q, p, g]) * π_grid[p, g])
    
    # CDF evaluated by trapezium rule for integration; 1/2 * (u(z_{i-1}) + u(z_i)) * Δx
    exp_fg = exp_fg[:, :, 2:end] + exp_fg[:, :, 1:end-1] .|> full_quant
    @tullio trapz[q, p, g] := (Δg[p, g] * exp_fg[q, p, g]) / 2
    return cumsum(trapz, dims=3), grid, st
end

function gausslegendre_quadrature(ebm, ps, st; ε::half_quant=eps(half_quant))
    """Gauss-Legendre quadrature for numerical integration"""

    # Map domains
    a, b = minimum(st[Symbol("1")].grid; dims=2), maximum(st[Symbol("1")].grid; dims=2)
    if b == ebm.fcns_qp[Symbol("1")].grid_size
        a, b = ebm.fcns_qp[Symbol("1")].grid_range
    end
    
    nodes = (a + b) ./ 2 .+ (b - a) ./ 2 .* device(ebm.nodes)
    weights = (b - a) ./ 2 .* device(ebm.weights)
    nodes_cpu = cpu_device()(nodes)

    π_nodes = ebm.prior_type == "lognormal" ? ebm.π_pdf(nodes, ε) : ebm.π_pdf(nodes)

    # Energy function of each component, q -> #
    nodes, st = prior_fwd(ebm, ps, st, nodes)

    # CDF evaluated by trapezium rule for integration; w_i * u(z_i)
    @tullio trapz[q, p, g] := (exp(nodes[q, p, g]) * π_nodes[p, g]) * weights[p, g]
    return cumsum(trapz .|> full_quant, dims=3), nodes_cpu, st
end

function sample_prior(
    ebm,
    num_samples::Int, 
    ps,
    st;
    seed::Int=1,
    ε::half_quant=eps(half_quant)
    )
    """
    Component-wise inverse transform sampling for the ebm-prior.
    p = components of model
    q = number of models

    Args:
        prior: The ebm-prior.
        ps: The parameters of the ebm-prior.
        st: The states of the ebm-prior.

    Returns:
        z: The samples from the ebm-prior, (num_samples, q). 
        seed: The updated seed.
    """

    cdf, grid, st = ebm.quad(ebm, ps, st; ε=ε)
    grid_size = size(grid, 2)
    cdf = cat(zeros(ebm.q_size, ebm.p_size, 1), cpu_device()(cdf), dims=3) # Add 0 to start of CDF

    seed, rng = next_rng(seed)
    rand_vals = rand(rng, full_quant, ebm.q_size, ebm.p_size, num_samples) .* cdf[:, :, end] 
    
    z = Array{full_quant}(undef, ebm.q_size, ebm.p_size, num_samples)
    Threads.@threads for q in 1:ebm.q_size
        for p in 1:ebm.p_size
            for b in 1:num_samples
                # First trapezium where CDF >= rand_val
                rv = rand_vals[q, p, b]
                idx = searchsortedfirst(cdf[q, p, :], rv) # Index of upper trapezium bound

                # Edge cases
                idx = idx == 1 ? 2 : idx
                idx = idx > grid_size ? grid_size : idx

                # Trapezium bounds
                z1, z2 = grid[p, idx-1], grid[p, idx] 
                cd1, cd2 = cdf[q, p, idx-1], cdf[q, p, idx]
 
                # Linear interpolation
                z[q, p, b] = z1 + (z2 - z1) * ((rv - cd1) / (cd2 - cd1))
            end
        end
    end

    return device(half_quant.(z)), st, seed
end

function log_prior(
    ebm, 
    z::AbstractArray{half_quant},
    ps, 
    st;
    ε::full_quant=eps(full_quant),
    )
    """
    Evaluate the unnormalized log-probability of the ebm-prior.
    The likelihood of samples from each model, z_qp, is evaluated .

    ∑_q [ ∑_p f_{q,p}(z_qp) ]
    
    Args:
        ebm: The ebm-prior.
        z: The component-wise latent samples to evaulate the measure on, (num_samples, q)
        ps: The parameters of the ebm-prior.
        st: The states of the ebm-prior.
        normalize: Whether to normalize the log-probability.
        ε: The small value to avoid log(0).
        agg: Whether to sum the log-probability over the samples.

    Returns:
        The unnormalized log-probability of the ebm-prior.
        The updated states of the ebm-prior.
    """

    log_p = zeros(size(z)[end]) |> device

    for q in 1:ebm.q_size
        f, st = prior_fwd(ebm, ps, st, z[q, :, :])
        log_p += dropdims(sum(f[q, :, :]; dims=1); dims=1)
    end

    return log_p, st
end

function init_ebm_prior(
    conf::ConfParse;
    prior_seed::Int=1,
    )
    widths = (
        try 
            parse.(Int, retrieve(conf, "EBM_PRIOR", "layer_widths"))
        catch
            parse.(Int, split(retrieve(conf, "EBM_PRIOR", "layer_widths"), ","))
        end
    )

    spline_degree = parse(Int, retrieve(conf, "EBM_PRIOR", "spline_degree"))
    layernorm = parse(Bool, retrieve(conf, "EBM_PRIOR", "layer_norm"))
    base_activation = retrieve(conf, "EBM_PRIOR", "base_activation")
    spline_function = retrieve(conf, "EBM_PRIOR", "spline_function")
    grid_size = parse(Int, retrieve(conf, "EBM_PRIOR", "grid_size"))
    grid_update_ratio = parse(half_quant, retrieve(conf, "EBM_PRIOR", "grid_update_ratio"))
    ε_scale = parse(half_quant, retrieve(conf, "EBM_PRIOR", "ε_scale"))
    μ_scale = parse(full_quant, retrieve(conf, "EBM_PRIOR", "μ_scale"))
    σ_base = parse(full_quant, retrieve(conf, "EBM_PRIOR", "σ_base"))
    σ_spline = parse(full_quant, retrieve(conf, "EBM_PRIOR", "σ_spline"))
    init_τ = parse(full_quant, retrieve(conf, "EBM_PRIOR", "init_τ"))
    τ_trainable = parse(Bool, retrieve(conf, "EBM_PRIOR", "τ_trainable"))
    τ_trainable = spline_function == "B-spline" ? false : τ_trainable
    prior_type = retrieve(conf, "EBM_PRIOR", "π_0")

    grid_range = Dict(
        "ebm" => parse.(half_quant, retrieve(conf, "EBM_PRIOR", "grid_range")),
        "lognormal" => [0,5] .|> half_quant,
        "gaussian" => [-1.5,1.5] .|> half_quant,
        "uniform" => [0,1] .|> half_quant,
    )[prior_type]

    eps = parse(half_quant, retrieve(conf, "TRAINING", "eps"))
    
    sample_function = (m, n, p, s, seed) -> @ignore_derivatives sample_prior(m, n, p, Lux.testmode(s); seed=seed, ε=eps)
    
    functions = NamedTuple()
    for i in eachindex(widths[1:end-1])
        prior_seed, rng = next_rng(prior_seed)
        base_scale = (μ_scale * (full_quant(1) / √(full_quant(widths[i])))
        .+ σ_base .* (randn(rng, full_quant, widths[i], widths[i+1]) .* full_quant(2) .- full_quant(1)) .* (full_quant(1) / √(full_quant(widths[i]))))

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
        init_τ=init_τ,
        τ_trainable=τ_trainable,
        )

        @reset functions[Symbol("$i")] = func

        if (layernorm && i < length(widths)-1)
            @reset functions[Symbol("ln_$i")] = Lux.LayerNorm(widths[i+1]) 
        end

    end


    quadrature_method = Dict(
        "gausslegendre" => gausslegendre_quadrature,
        "trapezium" => trapezium_quadrature
    )[retrieve(conf, "EBM_PRIOR", "quadrature_method")]
    N_quad = parse(Int, retrieve(conf, "EBM_PRIOR", "GaussQuad_nodes"))
    nodes, weights = gausslegendre(N_quad)
    nodes = repeat(nodes', first(widths), 1) .|> half_quant
    weights = full_quant.(weights)'

    return ebm_prior(functions, layernorm, length(widths)-1, prior_type, prior_pdf[prior_type], sample_function, first(widths), last(widths), quadrature_method, N_quad, nodes, weights)
end

function Lux.initialparameters(rng::AbstractRNG, prior::ebm_prior)
    ps = NamedTuple(Symbol("$i") => Lux.initialparameters(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)

    if prior.layernorm 
        for i in 1:prior.depth-1
            @reset ps[Symbol("ln_$i")] = Lux.initialparameters(rng, prior.fcns_qp[Symbol("ln_$i")]) 
        end
    end

    return ps 
end
 
function Lux.initialstates(rng::AbstractRNG, prior::ebm_prior)
    st = NamedTuple(Symbol("$i") => Lux.initialstates(rng, prior.fcns_qp[Symbol("$i")]) for i in 1:prior.depth)

    if prior.layernorm 
        for i in 1:prior.depth-1
            @reset st[Symbol("ln_$i")] = Lux.initialstates(rng, prior.fcns_qp[Symbol("ln_$i")]) |> hq
        end
    end

    return st 
end

end