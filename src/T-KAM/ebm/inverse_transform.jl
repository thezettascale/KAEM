
module InverseTransformSampling

export sample_univariate, sample_mixture, gausslegendre_quadrature, trapezium_quadrature

using CUDA, KernelAbstractions, LinearAlgebra, Random, Lux, LuxCUDA
using NNlib: softmax
using ChainRules: @ignore_derivatives

include("../../utils.jl")
include("log_prior_fcns.jl")
using .Utils: device, next_rng, half_quant, full_quant, fq
using .LogPriorFCNs: prior_fwd
using Flux: onehotbatch

function trapezium_quadrature(
    ebm, 
    ps, 
    st; 
    ε::T=eps(half_quant),
    component_mask::Union{AbstractArray{<:half_quant}, Nothing}=nothing
    ) where {T<:half_quant}
    """Trapezoidal rule for numerical integration"""

    # Evaluate prior on grid [0,1]
    f_grid = st[Symbol("1")].grid
    grid = f_grid |> cpu_device() 
    Δg = f_grid[:, 2:end] - f_grid[:, 1:end-1] 
    
    π_grid = ebm.prior_type == "learnable_gaussian" ? ebm.π_pdf(f_grid', ps, ε) : ebm.π_pdf(f_grid, ε)
    π_grid = ebm.prior_type == "learnable_gaussian" ? π_grid' : π_grid

    # Energy function of each component
    f_grid, st = prior_fwd(ebm, ps, st, f_grid)
    Q, P, G = size(f_grid)
   
    exp_fg = zeros(T, Q, P, G) |> device
    if component_mask !== nothing
        exp_fg = exp.(f_grid) .* reshape(π_grid, Q, 1, G)
        exp_fg = sum(reshape(exp_fg, Q, P, 1, G) .* reshape(component_mask, Q, P, B, 1), dims=2)
        exp_fg = dropdims(exp_fg, dims=2)
    else
        exp_fg = exp.(f_grid) .* reshape(π_grid, 1, P, G)
    end

    # CDF evaluated by trapezium rule for integration; 1/2 * (u(z_{i-1}) + u(z_i)) * Δx
    exp_fg = exp_fg[:, :, 2:end] + exp_fg[:, :, 1:end-1] 
    trapz = exp_fg .* reshape(Δg, 1, P, G-1) ./ 2
    return trapz, grid, st
end

function get_gausslegendre(ebm, ps, st)
    """Get Gauss-Legendre nodes and weights for prior's domain"""
    
    a, b = minimum(st[Symbol("1")].grid; dims=2), maximum(st[Symbol("1")].grid; dims=2)
    
    no_grid = (ebm.fcns_qp[Symbol("1")].spline_string == "FFT" || 
        ebm.fcns_qp[Symbol("1")].spline_string == "Cheby" ||
        ebm.fcns_qp[Symbol("1")].spline_string == "Gottlieb"
    )
    
    if no_grid
        a = fill(half_quant(first(ebm.fcns_qp[Symbol("1")].grid_range)), size(a)) |> device
        b = fill(half_quant(last(ebm.fcns_qp[Symbol("1")].grid_range)), size(b)) |> device
    end
    
    nodes = (a + b) ./ 2 .+ (b - a) ./ 2 .* device(ebm.nodes)
    weights = (b - a) ./ 2 .* device(ebm.weights)
    nodes_cpu = cpu_device()(nodes)
    
    return nodes, weights, nodes_cpu
end

function gausslegendre_quadrature(
    ebm, 
    ps, 
    st; 
    ε::T=eps(half_quant),
    component_mask::Union{AbstractArray{T}, Nothing}=nothing
    ) where {T<:half_quant}
    """Gauss-Legendre quadrature for numerical integration"""

    nodes, weights, nodes_cpu = @ignore_derivatives get_gausslegendre(ebm, ps, st)
    π_nodes = ebm.prior_type == "learnable_gaussian" ? ebm.π_pdf(nodes', ps, ε) : ebm.π_pdf(nodes, ε)
    π_nodes = ebm.prior_type == "learnable_gaussian" ? π_nodes' : π_nodes

    # Energy function of each component
    nodes, st = prior_fwd(ebm, ps, st, nodes)
    Q, P, G = size(nodes)

    # CDF evaluated by trapezium rule for integration; w_i * u(z_i)
    if component_mask !== nothing
        tmp = exp.(nodes) .* reshape(π_nodes, Q, 1, G)  # (Q, P, G)
        trapz = sum(
            reshape(tmp, Q, P, 1, G) .* reshape(component_mask, Q, P, size(component_mask, 3), 1),
            dims=2
        )
        trapz = dropdims(trapz, dims=2)  # (Q, B, G)
        trapz = trapz .* reshape(weights, Q, 1, G)  # (Q, B, G)
        return trapz, nodes_cpu, st
    else
        trapz = exp.(nodes) .* reshape(π_nodes, 1, P, G) .* reshape(weights, 1, P, G)  # (Q, P, G)
        return trapz, nodes_cpu, st
    end
end

function choose_component(
    α::AbstractArray{T}, 
    num_samples::Int, 
    q_size::Int, 
    p_size::Int; 
    seed::Int=1
    ) where {T<:half_quant}
    """
    Creates a one-hot mask for mixture model, q, to select one component, p.

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
    mask = Array{T}(undef, q_size, p_size, num_samples) 
    Threads.@threads for q in 1:q_size
        i = searchsortedfirst.(Ref(α[q, :]), rand_vals[q, :])
        replace!(i, p_size + 1 => p_size) # Edge case 
        mask[q, :, :] = onehotbatch(i, 1:p_size) .|> T
    end
    
    return mask |> device, seed
end

function sample_univariate(
    ebm,
    num_samples::Int, 
    ps,
    st;
    seed::Int=1,
    ε::T=eps(T)
    ) where {T<:half_quant}

    cdf, grid, st = ebm.quad(ebm, ps, st, nothing)
    grid_size = size(grid, 2)
    grid = grid .|> full_quant

    cdf = cat(
        zeros(full_quant, ebm.q_size, ebm.p_size, 1), # Add 0 to start of CDF
        cpu_device()(cumsum(cdf .|> full_quant; dims=3)), # Cumulative trapezium = CDF
        dims=3) 

    seed, rng = next_rng(seed)
    rand_vals = rand(rng, full_quant, 1, ebm.p_size, num_samples) .* cdf[:, :, end] 
    
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

    return device(T.(z)), st, seed
end

function sample_mixture(
    ebm,
    num_samples::Int, 
    ps,
    st;
    seed::Int=1,
    ε::T=eps(T)
    ) where {T<:half_quant}
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
    mask, seed = choose_component(
        ps[Symbol("α")],
        num_samples,
        ebm.q_size,
        ebm.p_size;
        seed=seed
    )

    cdf, grid, st = ebm.quad(ebm, ps, st, mask)
    grid_size = size(grid, 2)
    grid = grid .|> full_quant

    cdf = cat(
        zeros(full_quant, ebm.q_size, num_samples, 1), # Add 0 to start of CDF
        cpu_device()(cumsum(cdf .|> full_quant; dims=3)), # Cumulative trapezium = CDF
        dims=3) 

    seed, rng = next_rng(seed)
    rand_vals = rand(rng, full_quant, ebm.q_size, num_samples) .* cdf[:, :, end] 

    z = Array{full_quant}(undef, ebm.q_size, 1, num_samples)
    Threads.@threads for q in 1:ebm.q_size
        for b in 1:num_samples
            # First trapezium where CDF >= rand_val
            rv = rand_vals[q, b]
            idx = searchsortedfirst(cdf[q, b, :], rv) # Index of upper trapezium bound

            # Edge cases
            idx = idx == 1 ? 2 : idx
            idx = idx > grid_size ? grid_size : idx

            # Trapezium bounds
            z1, z2 = grid[q, idx-1], grid[q, idx] 
            cd1, cd2 = cdf[q, b, idx-1], cdf[q, b, idx]

            # Linear interpolation
            z[q, 1, b] = z1 + (z2 - z1) * ((rv - cd1) / (cd2 - cd1))
        end
    end

    return device(T.(z)), st, seed
end

end