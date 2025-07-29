module GridUpdating

export update_fcn_grid

using CUDA, Accessors, ComponentArrays, Lux, NNlib, LinearAlgebra, Random, LuxCUDA

using ..Utils
using ..UnivariateFunctions
using ..UnivariateFunctions.spline_functions

function update_fcn_grid(
    l::univariate_function{T,U},
    ps::ComponentArray{T},
    st::ComponentArray{T},
    x::AbstractArray{T};
    ε::T=full_quant(1f-3)
)::Tuple{AbstractArray{T},AbstractArray{U}} where {T<:half_quant,U<:full_quant}
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
    range = collect(T, 0:num_interval)[:, :] |> permutedims |> pu
    grid_uniform = h .* range .+ grid_adaptive[:, 1:1]

    # Grid is a convex combination of the uniform and adaptive grid
    grid = l.grid_update_ratio .* grid_uniform + (1 - l.grid_update_ratio) .* grid_adaptive
    new_grid = extend_grid(grid; k_extend = l.spline_degree)
    new_coef =
        l.spline_string == "FFT" ? curve2coef(l.basis_function, x_sort, y, new_grid, τ) :
        curve2coef(l.basis_function, x_sort, y, new_grid, τ; ε=U(ε))

    return new_grid, new_coef
end

end
