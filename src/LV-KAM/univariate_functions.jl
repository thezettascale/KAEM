module univariate_functions

export univariate_function, init_function, fwd, update_fcn_grid

using CUDA, KernelAbstractions, Tullio, Accessors
using Lux, NNlib, LinearAlgebra, Random, LuxCUDA

include("spline_bases.jl")
include("../utils.jl")
using .spline_functions
using .Utils: device, quant

SplineBasis_mapping = Dict(
    "B-spline" => B_spline_basis,
    "RBF" => RBF_basis,
    "RSWAF" => RSWAF_basis,
)

activation_mapping = Dict(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.tanh_fast,
    "sigmoid" => NNlib.sigmoid_fast,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu,
    "selu" => NNlib.selu,
    "tanh" => NNlib.tanh_fast,
    "silu" => x -> x .* NNlib.sigmoid_fast(x),
    "elu" => NNlib.elu,
    "celu" => NNlib.celu,
    "none" => x -> x .* 0f0
)

struct univariate_function <: Lux.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    spline_degree::Int
    base_activation::Function
    spline_function::Function
    grid::AbstractArray{quant}
    grid_size::Int
    grid_update_ratio::quant
    grid_range::Tuple{quant, quant}
    ε_scale::quant
    σ_base::AbstractArray{quant}
    σ_spline::quant
    init_η::AbstractArray{quant}
    η_trainable::Bool
end

function init_function(
    in_dim::Int,
    out_dim::Int;
    spline_degree::Int=3,
    base_activation::AbstractString="silu",
    spline_function::AbstractString="B-spline",
    grid_size::Int=5,
    grid_update_ratio::quant=2f-2,
    grid_range::Tuple{quant, quant}=(0f0, 1f0),
    ε_scale::quant=1f-1,
    σ_base::AbstractArray{quant}=[NaN32],
    σ_spline::quant=1f0,
    init_η::quant=1f0,
    η_trainable::Bool=true
)

    grid = quant.(range(grid_range[1], grid_range[2], length=grid_size + 1)) |> collect |> x -> reshape(x, 1, length(x)) |> device
    grid = repeat(grid, in_dim, 1) 
    grid = extend_grid(grid; k_extend=spline_degree)  
    σ_base = any(isnan.(σ_base)) ? ones(quant, in_dim, out_dim) : σ_base
    base_activation = get(activation_mapping, base_activation, x -> x .* NNlib.sigmoid_fast(x))
    spline_function = get(SplineBasis_mapping, spline_function, B_spline_basis)
    
    return univariate_function(in_dim, out_dim, spline_degree, base_activation, spline_function, grid, grid_size, grid_update_ratio, grid_range, ε_scale, σ_base, σ_spline, [init_η], η_trainable)
end

function Lux.initialparameters(rng::AbstractRNG, l::univariate_function)
    ε = ((rand(rng, quant, l.grid_size + 1, l.in_dim, l.out_dim) .- 0.5f0) .* l.ε_scale ./ l.grid_size) |> device
    coef = cpu_device()(curve2coef(l.grid[:, l.spline_degree+1:end-l.spline_degree] |> permutedims, ε, l.grid; k=l.spline_degree, scale=device(l.init_η)))
    w_base = glorot_normal(rng, quant, l.in_dim, l.out_dim) .* l.σ_base 
    w_sp = glorot_normal(rng, quant, l.in_dim, l.out_dim) .* l.σ_spline
    return l.η_trainable ? (w_base=w_base, w_sp=w_sp, coef=coef, basis_η=l.init_η) : (w_base=w_base, w_sp=w_sp, coef=coef)
end

function Lux.initialstates(rng::AbstractRNG, l::univariate_function)
    mask = ones(quant, l.in_dim, l.out_dim)
    return l.η_trainable ? (mask=mask) : (mask=mask, basis_η=l.init_η)
end

function fwd(l, ps, st, x)
    """
    Forward pass for the univariate function.

    Args:
        l: The univariate function layer.
        ps: The parameters of the layer.
        st: The states of the layer.
        x_p: The input, (b, i).

    Returns:
        The output, (b, i, o), containing all fcn_{q,p}(x_p)
    """

    w_base, w_sp, coef = ps.w_base, ps.w_sp, ps.coef
    mask = l.η_trainable ? st : st.mask
    η = l.η_trainable ? ps.basis_η : st.basis_η

    base = l.base_activation(x)
    y = coef2curve(x, l.grid, coef; k=l.spline_degree, scale=η)

    return @tullio out[b, i, o] := (w_base[i, o] * base[b, i] + w_sp[i, o] * y[b, i, o]) * mask[i, o]
end

function update_fcn_grid(l, ps, st, x)
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
    b_size = size(x, 1)
    coef = ps.coef
    η = l.η_trainable ? ps.basis_η : st.basis_η
    
    x_sort = sort(x, dims=1)
    current_splines = coef2curve(x_sort, l.grid, coef; k=l.spline_degree, scale=η)

    # Adaptive grid - concentrate grid points around regions of higher density
    num_interval = size(l.grid, 2) - 2*l.spline_degree - 1
    ids = [div(b_size * i, num_interval) + 1 for i in 0:num_interval-1]
    grid_adaptive = mapreduce(i -> view(x_sort, i:i, :), vcat, ids)
    grid_adaptive = vcat(grid_adaptive, x_sort[end:end, :])
    grid_adaptive = grid_adaptive |> permutedims 

    # Uniform grid
    h = (grid_adaptive[:, end:end] .- grid_adaptive[:, 1:1]) ./ num_interval # step size
    range = collect(quant, 0:num_interval)[:, :] |> permutedims |> device
    grid_uniform = h .* range .+ grid_adaptive[:, 1:1] 

    # Grid is a convex combination of the uniform and adaptive grid
    grid = l.grid_update_ratio .* grid_uniform + (1 - l.grid_update_ratio) .* grid_adaptive
    new_grid = extend_grid(grid; k_extend=l.spline_degree) 
    new_coef = curve2coef(x_sort, current_splines, new_grid; k=l.spline_degree, scale=η, basis_function=l.spline_function)

    return new_grid, new_coef
end

end