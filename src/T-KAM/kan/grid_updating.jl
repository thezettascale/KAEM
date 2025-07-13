module GridUpdating

export update_model_grid

using CUDA, KernelAbstractions, Accessors, Lux, NNlib, LinearAlgebra, Random, LuxCUDA

include("univariate_functions.jl")
include("../../utils.jl")
using .UnivariateFunctions: fwd, coef2curve, curve2coef, extend_grid
using .Utils: half_quant, device, next_rng

function update_fcn_grid(l, ps, st, x::AbstractArray{T}) where {T<:half_quant}
    """
    Adapt the function's grid to the distribution of the input data.

    Args:
        l: The univariate function layer.
        ps: The parameters of the layer.
        st: The state of the layer.
        x_p: The input of size (b, i).

    Returns:
        new_grid: The updated grid.
        new_coef: The updated spline coefficients.
    """
    sample_size = size(x, 2)
    coef = ps.coef
    τ = l.τ_trainable ? ps.basis_τ : st.basis_τ
    
    x_sort = sort(x, dims=2) 
    y = coef2curve(x_sort, st.grid, coef; k=l.spline_degree, scale=τ, basis_function=l.spline_function) .|> half_quant

    # Adaptive grid - concentrate grid points around regions of higher density
    num_interval = size(st.grid, 2) - 2*l.spline_degree - 1
    ids = [div(sample_size * i, num_interval) + 1 for i in 0:num_interval-1]'
    grid_adaptive = reduce(hcat, map(i -> view(x_sort, :, i:i), ids))
    grid_adaptive = hcat(grid_adaptive, x_sort[:, end:end])
    grid_adaptive = grid_adaptive  

    # Uniform grid
    h = (grid_adaptive[:, end:end] .- grid_adaptive[:, 1:1]) ./ num_interval # step size
    range = collect(T, 0:num_interval)[:, :] |> permutedims |> device
    grid_uniform = h .* range .+ grid_adaptive[:, 1:1] 

    # Grid is a convex combination of the uniform and adaptive grid
    grid = l.grid_update_ratio .* grid_uniform + (1 - l.grid_update_ratio) .* grid_adaptive
    new_grid = extend_grid(grid; k_extend=l.spline_degree)
    new_coef = curve2coef(x_sort, y, new_grid; k=l.spline_degree, scale=τ, basis_function=l.spline_function)

    return new_grid, new_coef
end

function update_model_grid(
    model::T_KAM,
    x::AbstractArray{T},
    ps, 
    st; 
    seed::Int=1
    )  where {T<:half_quant}
    """
    Update the grid of the likelihood model using samples from the prior.

    Args:
        model: The model.
        x: Data samples.
        ps: The parameters of the model.
        st: The states of the model.

    Returns:
        The updated model.
        The updated params.
        The updated seed.
    """
    ps = ps .|> T

    temps = model.N_t > 1 ? collect(T, [(k / model.N_t)^model.p[st.train_idx] for k in 0:model.N_t])[2:end] |> device : 0

    if model.update_prior_grid

        z, _, seed = ((model.MALA || model.N_t > 1) ? 
            model.posterior_sample(model, x, temps, ps, st, seed) : 
            model.prior.sample_z(model, model.grid_updates_samples, ps, st, seed)
            )

        P, Q = (model.MALA || model.N_t > 1) ? size(z)[1:2] : reverse(size(z)[1:2])
        z = reshape(z, P, Q, :)
        B = size(z, 3)
        z = reshape(z, P, Q*B)

        for i in 1:model.prior.depth
            new_grid, new_coef = update_fcn_grid(model.prior.fcns_qp[Symbol("$i")], ps.ebm[Symbol("$i")], st.ebm[Symbol("$i")], z)
            @reset ps.ebm[Symbol("$i")].coef = new_coef
            @reset st.ebm[Symbol("$i")].grid = new_grid

            z = fwd(model.prior.fcns_qp[Symbol("$i")], ps.ebm[Symbol("$i")], st.ebm[Symbol("$i")], z)
            z = i == 1 ? reshape(z, size(z, 2), :) : dropdims(sum(z, dims=1); dims=1)

            if model.prior.layernorm && i < model.prior.depth
                z, st_ebm = Lux.apply(model.prior.fcns_qp[Symbol("ln_$i")], z, ps.ebm[Symbol("ln_$i")], st.ebm[Symbol("ln_$i")])
                @reset st.ebm[Symbol("ln_$i")] = st_ebm
            end
        end
    end
        
    # Only update if KAN-type generator requires
    (!model.update_llhood_grid || model.lkhood.CNN || model.lkhood.seq_length > 1) && return model, T.(ps), st, seed

    z, _, seed = ((model.MALA || model.N_t > 1) ? 
        model.posterior_sample(model, x, temps, ps, st, seed) : 
        model.prior.sample_z(model, model.grid_updates_samples, ps, st, seed))

    z = dropdims(sum(reshape(z, size(z, 1), size(z, 2), :); dims=2); dims=2)

    for i in 1:model.lkhood.depth
        new_grid, new_coef = update_fcn_grid(model.lkhood.Φ_fcns[Symbol("$i")], ps.gen[Symbol("$i")], st.gen[Symbol("$i")], z)
        @reset ps.gen[Symbol("$i")].coef = new_coef
        @reset st.gen[Symbol("$i")].grid = new_grid

        z = fwd(model.lkhood.Φ_fcns[Symbol("$i")], ps.gen[Symbol("$i")], st.gen[Symbol("$i")], z)
        z = dropdims(sum(z, dims=1); dims=1)

        if model.lkhood.layernorm && i < model.lkhood.depth
            z, st_gen = Lux.apply(model.lkhood.Φ_fcns[Symbol("ln_$i")], z, ps.gen[Symbol("ln_$i")], st.gen[Symbol("ln_$i")])
            @reset st.gen[Symbol("ln_$i")] = st_gen
        end
    end

    return model, T.(ps), st, seed
end

end