module GridUpdating

export update_model_grid

using CUDA,
    KernelAbstractions,
    Accessors,
    ComponentArrays,
    Lux,
    NNlib,
    LinearAlgebra,
    Random,
    LuxCUDA

include("spline_bases.jl")
include("univariate_functions.jl")
include("../../utils.jl")
using .spline_functions: extend_grid
using .UnivariateFunctions
using .Utils: half_quant, full_quant, device, symbol_map

function update_fcn_grid(
    l,
    ps::ComponentArray{T},
    st::NamedTuple,
    x::AbstractArray{T},
)::Tuple{AbstractArray{T},AbstractArray{full_quant}} where {T<:half_quant}
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

    x_sort = sort(x, dims = 2)
    y = l.coef2curve(x_sort, st.grid, coef, τ) .|> half_quant

    # Adaptive grid - concentrate grid points around regions of higher density
    num_interval = size(st.grid, 2) - 2*l.spline_degree - 1
    ids = [div(sample_size * i, num_interval) + 1 for i = 0:(num_interval-1)]'
    grid_adaptive = reduce(hcat, map(i -> view(x_sort, :, i:i), ids))
    grid_adaptive = hcat(grid_adaptive, x_sort[:, end:end])
    grid_adaptive = grid_adaptive

    # Uniform grid
    h = (grid_adaptive[:, end:end] .- grid_adaptive[:, 1:1]) ./ num_interval # step size
    range = collect(T, 0:num_interval)[:, :] |> permutedims |> device
    grid_uniform = h .* range .+ grid_adaptive[:, 1:1]

    # Grid is a convex combination of the uniform and adaptive grid
    grid = l.grid_update_ratio .* grid_uniform + (1 - l.grid_update_ratio) .* grid_adaptive
    new_grid = extend_grid(grid; k_extend = l.spline_degree)
    new_coef = l.curve2coef(x_sort, y, new_grid, τ)

    return new_grid, new_coef
end

function update_model_grid(
    model,
    x::AbstractArray{T},
    ps::ComponentArray{T},
    st::NamedTuple,
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{Any,ComponentArray{T},NamedTuple} where {T<:half_quant}
    """
    Update the grid of the likelihood model using samples from the prior.

    Args:
        model: The model.
        x: Data samples.
        ps: The parameters of the model.
        st: The states of the model.
        rng: The random number generator.

    Returns:
        The updated model.
        The updated params.
    """
    ps = ps .|> T

    temps =
        model.N_t > 1 ?
        collect(T, [(k / model.N_t)^model.p[st.train_idx] for k = 0:model.N_t])[2:end] :
        [one(T)]

    if model.update_prior_grid

        z, _ = (
            (model.MALA || model.N_t > 1) ?
            model.posterior_sample(model, x, temps, ps, st, rng) :
            model.prior.sample_z(model, model.grid_updates_samples, ps, st, rng)
        )

        Q, P = (
            (model.prior.ula || model.prior.mixture_model) ? reverse(size(z)[1:2]) :
            size(z)[1:2]
        )
        z = reshape(z, P, Q, :)
        B = size(z, 3)
        z = reshape(z, P, Q*B)

        for i = 1:model.prior.depth
            new_grid, new_coef = update_fcn_grid(
                model.prior.fcns_qp[i],
                ps.ebm.fcn[symbol_map[i]],
                st.ebm.fcn[symbol_map[i]],
                z,
            )
            @reset ps.ebm.fcn[symbol_map[i]].coef = new_coef
            @reset st.ebm.fcn[symbol_map[i]].grid = new_grid

            z, _ = model.prior.Lux.apply(prior.fcns_qp[i], z, ps.ebm.fcn[symbol_map[i]], st.ebm.fcn[symbol_map[i]])
            z = i == 1 ? reshape(z, size(z, 2), :) : dropdims(sum(z, dims = 1); dims = 1)

            if model.prior.layernorm_bool && i < model.prior.depth
                z, st_ebm = Lux.apply(
                    model.prior.layernorms[i],
                    z,
                    ps.ebm.layernorm[symbol_map[i]],
                    st.ebm.layernorm[symbol_map[i]],
                )
                @reset st.ebm.layernorm[symbol_map[i]] = st_ebm
            end
        end
    end

    # Only update if KAN-type generator requires
    (!model.update_llhood_grid || model.lkhood.CNN || model.lkhood.seq_length > 1) &&
        return model, T.(ps), st

    z, _ = (
        (model.MALA || model.N_t > 1) ?
        model.posterior_sample(model, x, temps, ps, st, rng) :
        model.prior.sample_z(model, model.grid_updates_samples, ps, st, rng)
    )

    z = dropdims(sum(reshape(z, size(z, 1), size(z, 2), :); dims = 2); dims = 2)

    for i = 1:model.lkhood.depth
        new_grid, new_coef = update_fcn_grid(
            model.lkhood.Φ_fcns[i],
            ps.gen.fcn[symbol_map[i]],
            st.gen.fcn[symbol_map[i]],
            z,
        )
        @reset ps.gen.fcn[symbol_map[i]].coef = new_coef
        @reset st.gen.fcn[symbol_map[i]].grid = new_grid

        z, _ = model.lkhood.Φ_fcns[i](z, ps.gen.fcn[symbol_map[i]], st.gen.fcn[symbol_map[i]])
        z = dropdims(sum(z, dims = 1); dims = 1)

        if model.lkhood.layernorm_bool && i < model.lkhood.depth
            z, st_gen = Lux.apply(
                model.lkhood.layernorms[i],
                z,
                ps.gen.layernorm[symbol_map[i]],
                st.gen.layernorm[symbol_map[i]],
            )
            @reset st.gen.layernorm[symbol_map[i]] = st_gen
        end
    end

    return model, T.(ps), st
end

end
