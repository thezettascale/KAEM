module InverseSampling

export sample_prior, prior_fwd

using CUDA, KernelAbstractions, Tullio
using Random, Distributions, Lux, LuxCUDA, LinearAlgebra, Accessors
using Flux: onehotbatch

include("../utils.jl")
include("univariate_functions.jl")
using .Utils: device, next_rng, removeZero, half_quant, full_quant, removeNeg
using .univariate_functions: fwd

function choose_component(
    α::AbstractArray{half_quant}, 
    num_samples::Int, 
    q_size::Int, 
    p_size::Int; 
    seed::Int=1
    )
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
    rand_vals = rand(rng, full_quant, q_size, num_samples) 
    α = cumsum(softmax(α .|> full_quant; dims=2); dims=2) |> cpu_device() 

    # Find the index of the first cdf value greater than the random value
    mask = Array{half_quant}(undef, q_size, p_size, num_samples) 
    Threads.@threads for q in 1:q_size
        i = searchsortedfirst.(Ref(α[q, :]), rand_vals[q, :])
        replace!(i, p_size + 1 => p_size) # Edge case 
        mask[q, :, :] = onehotbatch(i, 1:p_size) .|> half_quant
    end

    return mask |> device, seed
end

function prior_fwd(mix, ps, st, z::AbstractArray{half_quant})
    for i in 1:mix.depth
        z = fwd(mix.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = i == 1 ? reshape(z, size(z, 2), mix.q_size*size(z, 3)) : dropdims(sum(z, dims=1); dims=1)

        if mix.layernorm && i < mix.depth
            z, st_new = Lux.apply(mix.fcns_qp[Symbol("ln_$i")], z, ps[Symbol("ln_$i")], st[Symbol("ln_$i")])
            @reset st[Symbol("ln_$i")] = st_new
        end
    end
    return reshape(z, mix.q_size, mix.p_size, :), st
end

function trapezium_quadrature(mix, ps, st, component_mask::AbstractArray{half_quant})
    """Trapezoidal rule for numerical integration"""

    # Evaluate prior on grid [0,1]
    f_grid = st[Symbol("1")].grid
    grid = f_grid |> cpu_device() .|> full_quant
    Δg = f_grid[:, 2:end] - f_grid[:, 1:end-1] .|> full_quant
    
    π_grid = mix.prior_type == "lognormal" ? mix.π_pdf(f_grid, ε) : mix.π_pdf(f_grid)
    grid_size = size(f_grid, 2)

    # Energy function of each component, q -> p
    f_grid, st = prior_fwd(mix, ps, st, f_grid)

    # Filter out components
    @tullio exp_fg[q, g, b] := (exp(f_grid[q, p, g]) * π_grid[q, g]) * component_mask[q, p, b]
    exp_fg = exp_fg .|> full_quant

    # CDF evaluated by trapezium rule for integration; 1/2 * (u(z_{i-1}) + u(z_i)) * Δx
    trapz = (Δg .* (exp_fg[:, 2:end, :] + exp_fg[:, 1:end-1, :])) ./ 2

    return cumsum(trapz, dims=2) , grid, st
end

function gausslegendre_quadrature(mix, ps, st, component_mask::AbstractArray{half_quant}; ε::half_quant=eps(half_quant))
    """Gauss-Legendre quadrature for numerical integration"""

    # Map domains
    a, b = minimum(st[Symbol("1")].grid), maximum(st[Symbol("1")].grid)
    if b == mix.fcns_qp[Symbol("1")].grid_size
        a, b = mix.fcns_qp[Symbol("1")].grid_range
    end
    
    nodes = (a + b) ./ 2 .+ (b - a) ./ 2 .* mix.nodes 
    weights = (b - a) ./ 2 .* mix.weights |> device
    f_nodes = device(copy(nodes))

    π_nodes = mix.prior_type == "lognormal" ? mix.π_pdf(f_nodes, ε) : mix.π_pdf(f_nodes)
    π_nodes = permutedims(π_nodes[:,:,:], (1, 3, 2))
    component_mask = permutedims(component_mask[:,:,:,:], (1, 2, 4, 3))

    # Energy function of each component, q -> #
    f_nodes, st = prior_fwd(mix, ps, st, f_nodes)

    # Filter out components
    f_nodes = dropdims(sum(exp.(f_nodes) .* component_mask .* π_nodes, dims=2); dims=2)
    f_nodes = f_nodes .|> full_quant    
    
    # CDF evaluated by trapezium rule for integration; w_i * u(z_i)
    return cumsum(f_nodes .* weights, dims=2) , nodes, st
end

function sample_prior(
    mix,
    num_samples::Int, 
    ps,
    st;
    seed::Int=1,
    ε::half_quant=eps(half_quant)
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

    # Categorical component selection (per sample, per outer sum dimension)
    component_mask, seed = choose_component(
        ps[Symbol("α")],
        num_samples,
        mix.q_size,
        mix.p_size;
        seed=seed
    )

    cdf, grid, st = mix.quadrature_method == "trapezium" ? trapezium_quadrature(mix, ps, st, component_mask) : gausslegendre_quadrature(mix, ps, st, component_mask; ε=ε)
    grid_size = size(grid, 2)
    cdf = cat(zeros(mix.q_size, 1, num_samples), cpu_device()(cdf), dims=2) # Add 0 to start of CDF

    seed, rng = next_rng(seed)
    rand_vals = rand(rng, full_quant, mix.q_size, num_samples) .* cdf[:, end, :] 
    
    z = Array{full_quant}(undef, mix.q_size, num_samples)
    Threads.@threads for q in 1:mix.q_size
        for b in 1:num_samples
            # First trapezium where CDF >= rand_val
            rv = rand_vals[q, b]
            idx = searchsortedfirst(cdf[q, :, b], rv) # Index of upper trapezium bound

            # Edge cases
            idx = idx == 1 ? 2 : idx
            idx = idx > grid_size ? grid_size : idx

            # Trapezium bounds
            z1, z2 = grid[q, idx-1], grid[q, idx] 
            cd1, cd2 = cdf[q, idx-1, b], cdf[q, idx, b]

            # Linear interpolation
            z[q, b] = z1 + (z2 - z1) * ((rv - cd1) / (cd2 - cd1))
        end
    end

    return device(half_quant.(z)), st, seed
end

end