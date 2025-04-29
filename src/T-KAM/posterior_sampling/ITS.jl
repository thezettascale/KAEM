module LangevinSampling

export langevin_sampler, cross_entropy, l2

using CUDA, KernelAbstractions, Tullio, LinearAlgebra, Random, Lux, LuxCUDA, Distributions, Accessors, Statistics
using Zygote: gradient

include("../../utils.jl")
include("../EBM_prior.jl")
include("../KAN_likelihood.jl")
using .Utils: device, next_rng, half_quant, full_quant, fq
using .ebm_ebm_prior: log_prior, prior_fwd
using .KAN_likelihood: log_likelihood

function cross_entropy(x::AbstractArray{half_quant}, y::AbstractArray{half_quant}; ε::half_quant=eps(half_quant))
    return log.(x .+ ε) .* y ./ size(x, 1)
end

function l2(x::AbstractArray{half_quant}, y::AbstractArray{half_quant}; ε::half_quant=eps(half_quant))
    return -(x - y).^2
end

function trapezium_quadrature(m, ps, st, x::AbstractArray{half_quant}, t_k::half_quant; ε::half_quant=eps(half_quant), seed::Int=1)
    """Trapezoidal rule for numerical integration"""

    # Evaluate prior on grid [0,1]
    grid_gpu = st.ebm[Symbol("1")].grid
    Δg = grid_gpu[:, 2:end] - grid_gpu[:, 1:end-1] 
    
    π_grid = m.prior.prior_type == "lognormal" ? m.prior.π_pdf(grid_gpu, ε) : m.prior.π_pdf(grid_gpu)
    grid_size = size(grid_gpu, 2)

    # Energy function of each component
    g_ll = repeat(reshape(device(grid_gpu), 1, m.prior.p_size, size(grid_gpu, 2)), m.prior.q_size, 1, 1)
    lp, st_ebm = prior_fwd(m.prior, ps.ebm, st.ebm, grid_gpu)
    ll, st_gen, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, g_ll; seed=seed, ε=m.ε)
    @reset st.ebm = st_ebm
    @reset st.gen = st_gen

    # CDF evaluated by trapezium rule for integration; 1/2 * (u(z_{i-1}) + u(z_i)) * Δx
    @tullio exp_fg[q, p, s, g] := (exp(lp[q, p, g] + t_k * ll[s, g]) * π_grid[p, g])
    exp_fg = exp_fg[:, :, :, 2:end] + exp_fg[:, :, :, 1:end-1] 
    @tullio trapz[q, p, s, g] := (Δg[p, g] * exp_fg[q, p, s, g]) / 2
    return trapz, cpu_device()(grid_gpu), st, seed
end

function get_gausslegendre(ebm, ps, st)
    """Get Gauss-Legendre nodes and weights for prior's domain"""
    
    a, b = minimum(st[Symbol("1")].grid; dims=2), maximum(st[Symbol("1")].grid; dims=2)
    if any(b .== ebm.fcns_qp[Symbol("1")].grid_size)
        a = fill(half_quant(first(ebm.fcns_qp[Symbol("1")].grid_range)), size(a)) |> device
        b = fill(half_quant(last(ebm.fcns_qp[Symbol("1")].grid_range)), size(b)) |> device
    end
    
    nodes = (a + b) ./ 2 .+ (b - a) ./ 2 .* device(ebm.nodes)
    weights = (b - a) ./ 2 .* device(ebm.weights)
    nodes_cpu = cpu_device()(nodes)
    
    return nodes, weights, nodes_cpu
end

function gausslegendre_quadrature(m, ps, st, x::AbstractArray{half_quant}, t_k::half_quant; ε::half_quant=eps(half_quant), seed::Int=1)
    """Gauss-Legendre quadrature for numerical integration"""

    nodes, weights, nodes_cpu = get_gausslegendre(m.prior, ps.ebm, st.ebm)
    π_nodes = m.prior.prior_type == "lognormal" ? m.prior.π_pdf(nodes, ε) : m.prior.π_pdf(nodes)

    # Energy function of each component
    g_ll = repeat(reshape(device(nodes), 1, m.prior.p_size, size(nodes, 2)), m.prior.q_size, 1, 1)
    lp, st_ebm = prior_fwd(m.prior, ps.ebm, st.ebm, nodes)
    ll, st_gen, seed = log_likelihood(m.lkhood, ps.gen, st.gen, x, g_ll; seed=seed, ε=m.ε)
    @reset st.ebm = st_ebm
    @reset st.gen = st_gen

    # CDF evaluated by trapezium rule for integration; w_i * u(z_i)
    @tullio trapz[q, p, s, g] := (exp(lp[q, p, g] + t_k * ll[s, g]) * π_nodes[p, g]) * weights[p, g]
    return trapz, nodes_cpu, st, seed
end

function langevin_sampler(
    m,
    ps,
    st,
    x::AbstractArray{half_quant};
    t::AbstractArray{half_quant}=[half_quant(1)],
    N::Int=20,
    N_unadjusted::Int=1,
    Δη::full_quant=full_quant(2),
    η_min::full_quant=full_quant(1e-5),
    η_max::full_quant=full_quant(1),
    seed::Int=1,
    )
    """
    Don't be fooled by the name, this is Inverse Transform Sampling.

    Args:
        m: The model.
        ps: The parameters of the model.
        st: The states of the model.
        x: The data
        t: The temperatures if using Thermodynamic Integration.
        N: The number of iterations.
        η_init: The initial step size.
        seed: The seed for the random number generator.

    Returns:
        The posterior samples.
        The updated seed.
    """
    
    # Initialize from prior (already in bounded space)
    z, st_ebm, seed = m.prior.sample_z(m.prior, size(x)[end], ps.ebm, st.ebm, seed)
    @reset st.ebm = st_ebm
    z = z .|> full_quant |> cpu_device()

    T, S = length(t), size(x)[end]
    output = reshape(z, m.prior.q_size, m.prior.p_size, S, 1)

    k = 1

    z = Array{full_quant}(undef, m.prior.q_size, m.prior.p_size, S, T)
    while k < T + 1
        cdf, grid, st, seed = m.prior.quad_type == "trapezium" ? trapezium_quadrature(m, ps, st, x, t[k]; ε=m.ε, seed=seed) : gausslegendre_quadrature(m, ps, st, x, t[k]; ε=m.ε, seed=seed)

        grid = grid .|> full_quant
        grid_size = size(grid, 2)
        cdf = cat(
            zeros(full_quant, m.prior.q_size, m.prior.p_size, S, 1), # Add 0 to start of CDF
            cpu_device()(cumsum(cdf .|> full_quant; dims=4)), # Cumulative trapezium = CDF
            dims=4) 

        seed, rng = next_rng(seed)
        rand_vals = rand(rng, full_quant, 1, m.prior.p_size, S) .* cdf[:, :, :, end] 
        
        Threads.@threads for q in 1:m.prior.q_size
            for p in 1:m.prior.p_size
                for s in 1:S
                    rv = rand_vals[q, p, s]
                    idx = searchsortedfirst(cdf[q, p, s, :], rv)

                        idx = idx == 1 ? 2 : idx
                        idx = idx > grid_size ? grid_size : idx

                    z1, z2 = grid[p, idx-1], grid[p, idx]
                    cd1, cd2 = cdf[q, p, s, idx-1], cdf[q, p, s, idx]

                    z[q, p, s, k] = z1 + (z2 - z1) * ((rv - cd1) / (cd2 - cd1))
                end
            end
        end
        k += 1
    end
    return half_quant.(cat(output, z, dims=4)), st, seed
end
end