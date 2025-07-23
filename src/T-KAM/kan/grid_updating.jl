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
using .spline_functions: extend_grid, coef2curve_FFT, coef2curve_Spline, curve2coef
using .UnivariateFunctions
using .Utils: half_quant, full_quant, device, symbol_map

function update_fcn_grid(
    l,
    ps::ComponentArray{T},
    st::ComponentArray{T},
    x::AbstractArray{T},
)::Tuple{AbstractArray{T},AbstractArray{full_quant}} where {T<:half_quant}
    """
    Adapt the function's grid to the distribution of the input data.

    Args:
        l: The univariate function layer.
        ps: The parameters of the layer.
        st: The state of the KAN layer.
        x_p: The input of size (i, num_samples).

    Returns:
        new_grid: The updated grid.
        new_coef: The updated spline coefficients.
    """
    sample_size = size(x, 2)
    coef = ps.coef
    τ = l.τ_trainable ? ps.basis_τ : st.basis_τ

    x_sort = sort(x, dims = 2)
    y =
        l.spline_string == "FFT" ?
        coef2curve_FFT(l.basis_function, x_sort, st.grid, coef, τ) :
        coef2curve_Spline(l.basis_function, x_sort, st.grid, coef, τ) .|> half_quant

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
    new_coef =
        l.spline_string == "FFT" ? curve2coef(l.basis_function, x_sort, y, new_grid, τ) :
        curve2coef(l.basis_function, x_sort, y, new_grid, τ)

    return new_grid, new_coef
end

function update_model_grid(
    model,
    x::AbstractArray{T},
    ps::ComponentArray{T},
    kan_st::ComponentArray{T},
    st_lux::NamedTuple;
    temps::AbstractArray{T} = [one(T)],
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{Any,ComponentArray{T},ComponentArray{T},NamedTuple} where {T<:half_quant}
    """
    Update the grid of the likelihood model using samples from the prior.

    Args:
        model: The model.
        x: Data samples.
        ps: The parameters of the model.
        kan_st: The states of the KAN model.
        st_lux: The states of the Lux model.
        temps: The temperatures for thermodynamic models.
        rng: The random number generator.

    Returns:
        The updated model.
        The updated params.
        The updated KAN states.
        The updated Lux states. 
    """

    z = nothing
    sampled_bool = false
    if model.update_prior_grid

        if model.N_t > 1
            z = first(
                model.posterior_sample(model, x, temps[2:end], ps, kan_st, st_lux, rng),
            )
        elseif model.prior.ula || model.MALA
            z = first(model.posterior_sample(model, x, ps, kan_st, st_lux, rng))
        else
            z = first(
                model.prior.sample_z(
                    model,
                    model.grid_updates_samples,
                    ps,
                    kan_st,
                    st_lux,
                    rng,
                ),
            )
        end

        sampled_bool = true

        # If Cheby or FFT, need to update domain for inverse transform sampling
        if model.prior.fcns_qp[1].spline_string == "FFT" ||
           model.prior.fcns_qp[1].spline_string == "Cheby"
            if (model.MALA || model.N_t > 1 || model.prior.ula)
                new_domain = (minimum(z), maximum(z))
                @reset model.prior.fcns_qp[1].grid_range = new_domain
            end

            # Otherwise use KAN grid updating
        else
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
                    kan_st.ebm[symbol_map[i]],
                    z,
                )
                @reset ps.ebm.fcn[symbol_map[i]].coef = new_coef
                @reset kan_st.ebm[symbol_map[i]].grid = new_grid

                scale = (maximum(new_grid) - minimum(new_grid)) / (size(new_grid, 2) - 1)
                model.prior.fcns_qp[i].spline_string == "RBF" &&
                    @reset model.prior.fcns_qp[i].basis_function.scale = scale

                z = model.prior.fcns_qp[i](
                    z,
                    ps.ebm.fcn[symbol_map[i]],
                    kan_st.ebm[symbol_map[i]],
                )
                z =
                    i == 1 ? reshape(z, size(z, 2), :) :
                    dropdims(sum(z, dims = 1); dims = 1)

                if model.prior.layernorm_bool && i < model.prior.depth
                    z, st_ebm = Lux.apply(
                        model.prior.layernorms[i],
                        z,
                        ps.ebm.layernorm[symbol_map[i]],
                        st_lux.ebm.layernorm[symbol_map[i]],
                    )
                    @reset st_lux.ebm.layernorm[symbol_map[i]] = st_ebm
                end
            end
        end
    end

    # Only update if KAN-type generator requires
    (!model.update_llhood_grid || model.lkhood.CNN || model.lkhood.SEQ) &&
        return model, T.(ps), kan_st, st_lux

    if !sampled_bool
        if model.N_t > 1
            z = first(
                model.posterior_sample(model, x, temps[2:end], ps, kan_st, st_lux, rng),
            )
        elseif model.prior.ula || model.MALA
            z = first(model.posterior_sample(model, x, ps, kan_st, st_lux, rng))
        else
            z = first(
                model.prior.sample_z(
                    model,
                    model.grid_updates_samples,
                    ps,
                    kan_st,
                    st_lux,
                    rng,
                ),
            )
        end
    end

    z = dropdims(sum(reshape(z, size(z, 1), size(z, 2), :); dims = 2); dims = 2)

    for i = 1:model.lkhood.generator.depth
        new_grid, new_coef = update_fcn_grid(
            model.lkhood.generator.Φ_fcns[i],
            ps.gen.fcn[symbol_map[i]],
            kan_st.gen[symbol_map[i]],
            z,
        )
        @reset ps.gen.fcn[symbol_map[i]].coef = new_coef
        @reset kan_st.gen[symbol_map[i]].grid = new_grid

        scale = (maximum(new_grid) - minimum(new_grid)) / (size(new_grid, 2) - 1)
        model.lkhood.generator.Φ_fcns[i].spline_string == "RBF" &&
            @reset model.lkhood.generator.Φ_fcns[i].basis_function.scale = scale

        z = model.lkhood.generator.Φ_fcns[i](
            z,
            ps.gen.fcn[symbol_map[i]],
            kan_st.gen[symbol_map[i]],
        )
        z = dropdims(sum(z, dims = 1); dims = 1)

        if model.lkhood.generator.layer_norm_bool && i < model.lkhood.generator.depth
            z, st_gen = Lux.apply(
                model.lkhood.generator.layernorms[i],
                z,
                ps.gen.layernorm[symbol_map[i]],
                st_lux.gen.layernorm[symbol_map[i]],
            )
            @reset st_lux.gen.layernorm[symbol_map[i]] = st_gen
        end
    end

    return model, T.(ps), kan_st, st_lux
end

end
