module prior_sampler

export select_category, select_category_differentiable, sample_prior

using CUDA, KernelAbstractions, Tullio
using Lux, LinearAlgebra, Random, LuxCUDA, ComponentArrays
using Distributions: Categorical, Uniform
using Flux: onehotbatch
using NNlib: softmax, logsoftmax, sigmoid_fast

include("univariate_functions.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng

function select_category(
    α::AbstractArray, 
    τ::Float32, 
    in_dim::Int, 
    out_dim::Int, 
    num_samples::Int
    )

    α = cpu_device()(softmax(α))
    rand_vals = rand(Categorical(α), out_dim, num_samples) 
    return permutedims(collect(Float32, onehotbatch(rand_vals, 1:in_dim)), [3, 1, 2]) |> device 
end

function select_category_differentiable(
    α::AbstractArray, 
    τ::Float32, 
    in_dim::Int, 
    out_dim::Int, 
    num_samples::Int
    )

    α = logsoftmax(α)
    gumbel = -log.(-log.(rand(Uniform(0,1), num_samples, length(α), out_dim) .+ eps(Float32))) |> device
    logits = @tullio gumbel_logits[b, i, o] := α[i] + gumbel[b, i, o]
    return softmax(logits ./ τ, dims=2)
end

function sample_prior(
    prior, 
    num_samples::Int, 
    ps,
    st;
    init_seed::Int=1
    )
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

    previous_samples = zeros(Float32, num_samples) |> device
    sample_mask = zeros(Float32, num_samples) |> device
    seed = next_rng(init_seed)
    p_size = prior.fcns_qp[Symbol("1")].in_dim
    q_size = prior.fcns_qp[Symbol("$(prior.depth)")].out_dim

    # Categorical component selection (per sample, per outer sum dimension)
    chosen_components = prior.categorical_mask(
        ps.α, 
        prior.τ, 
        p_size, 
        q_size, 
        num_samples
        )

    # Rejection sampling
    while any(sample_mask .< 5f-1)

        # Draw candidate samples from proposal, i.e. prior
        seed = next_rng(seed) 
        z_p = rand(prior.π_0, num_samples, p_size) |> device # z ~ Q(z)

        # Forward pass of proposal samples through mixture model
        fz_qp = z_p
        for i in 1:prior.depth
            fz_qp = fwd(prior.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], fz_qp)
            fz_qp = i == 1 ? reshape(fz_qp, :, size(fz_qp, 3)) : sum(fz_qp, dims=2)[:, 1, :] 
        end
        fz_qp = reshape(fz_qp, size(z_p)..., q_size)

        # Forward pass of grid [0,1] through model
        f_grid = prior.fcns_qp[Symbol("1")].grid'
        for i in 1:prior.depth
            f_grid = fwd(prior.fcns_qp[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], f_grid)
            f_grid = i == 1 ? reshape(f_grid, :, size(f_grid, 3)) : sum(f_grid, dims=2)[:, 1, :] 
        end
        f_grid = reshape(f_grid, :, p_size, q_size)
        
        # Filter chosen components of mixture model, (samples x q)
        z_p = @tullio chosen[b, o] := z_p[b, i] * chosen_components[b, i, o]
        fz_qp = sum(fz_qp .* chosen_components, dims=2)[:,1,:] 

        # Grid search for max_z[ f_{q,c}(z) ] for chosen components
        f_grid = @tullio fg[b, g, i, o] := f_grid[g ,i, o]  * chosen_components[b, i, o]
        f_grid = prior.max_fcn(sum(f_grid; dims=3); dims=2)[:,1,1,:] # Filtered max f_qp, (samples x q)

        # Accept or reject
        seed = next_rng(seed)
        u_threshold = rand(Uniform(0,1), num_samples, q_size) |> device # u ~ U(0,1)
        accept_mask = prior.acceptance_fcn(u_threshold, fz_qp, f_grid) # Acceptance mask

        # Update samples
        previous_samples = z_p .* accept_mask .* (1 .- sample_mask) .+ previous_samples .* sample_mask
        sample_mask = accept_mask .* (1 .- sample_mask) .+ sample_mask 
    end

    return previous_samples, seed
end

end