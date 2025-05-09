module univariate_functions

export univariate_function, init_function, fwd, update_fcn_grid, activation_mapping

using CUDA, KernelAbstractions, Tullio, Accessors
using Lux, NNlib, LinearAlgebra, Random, LuxCUDA

include("spline_bases.jl")
include("../utils.jl")
using .spline_functions
using .Utils: device, half_quant, full_quant

SplineBasis_mapping = Dict(
    "B-spline" => B_spline_basis,
    "RBF" => RBF_basis,
    "RSWAF" => RSWAF_basis,
    "FFT" => FFT_basis,
    "Cheby" => Cheby_basis,
    "Gottlieb" => Gottlieb_basis
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
    "none" => x -> x .* zero(half_quant)
)

struct univariate_function{T<:half_quant, U<:full_quant} <: Lux.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    spline_degree::Int
    base_activation::Function
    spline_function::Function
    spline_string::String
    init_grid::AbstractArray{T}
    grid_size::Int
    grid_update_ratio::T
    grid_range::Tuple{T, T}
    ε_scale::T
    σ_base::AbstractArray{U}
    σ_spline::U
    init_τ::AbstractArray{U}
    τ_trainable::Bool
end

function init_function(
    in_dim::Int,
    out_dim::Int;
    spline_degree::Int=3,
    base_activation::AbstractString="silu",
    spline_function::AbstractString="B-spline",
    grid_size::Int=5,
    grid_update_ratio::T=half_quant(0.02),
    grid_range::Tuple{T, T}=(zero(half_quant), one(half_quant)),
    ε_scale::T=half_quant(0.1),
    σ_base::AbstractArray{U}=[full_quant(NaN)],
    σ_spline::U=one(full_quant),
    init_τ::U=one(full_quant),
    τ_trainable::Bool=true
) where {T<:half_quant, U<:full_quant}
    spline_degree = spline_function == "B-spline" ? spline_degree : 0
    grid_size = (spline_function == "Cheby" || spline_function == "Gottlieb") ? 1 : grid_size
    grid = spline_function == "FFT" ? collect(T, 0:grid_size) : range(grid_range[1], grid_range[2], length=grid_size + 1)
    grid = T.(grid) |> collect |> x -> reshape(x, 1, length(x)) |> device
    grid = repeat(grid, in_dim, 1) 
    grid = extend_grid(grid; k_extend=spline_degree) 
    σ_base = any(isnan.(σ_base)) ? ones(U, in_dim, out_dim) : σ_base
    base_activation = get(activation_mapping, base_activation, x -> x .* NNlib.sigmoid_fast(x))
    
    return univariate_function(
        in_dim, 
        out_dim, 
        spline_degree, 
        base_activation, 
        get(SplineBasis_mapping, spline_function, B_spline_basis), 
        spline_function,
        grid, grid_size, 
        grid_update_ratio, 
        grid_range, 
        ε_scale,
        σ_base, 
        σ_spline, 
        [init_τ], 
        τ_trainable
        )
end

function Lux.initialparameters(rng::AbstractRNG, l::univariate_function)
    
    w_base = glorot_normal(rng, full_quant, l.in_dim, l.out_dim) .* l.σ_base 
    w_sp = glorot_normal(rng, full_quant, l.in_dim, l.out_dim) .* l.σ_spline
    
    coef = nothing
    if l.spline_string == "FFT" 
        grid_norm_factor = collect(1:l.grid_size+1) .^ 2
        coef = glorot_normal(rng, full_quant, 2, l.in_dim, l.out_dim, l.grid_size+1) ./ (sqrt(l.in_dim) .* permutedims(grid_norm_factor[:,:,:,:], [2, 3, 4, 1])) 
    elseif !(l.spline_string == "Cheby" || l.spline_string == "Gottlieb")
        ε = ((rand(rng, half_quant, l.in_dim, l.out_dim, l.grid_size + 1) .- half_quant(0.5)) .* l.ε_scale ./ l.grid_size) |> device  
        coef = cpu_device()(curve2coef(l.init_grid[:, l.spline_degree+1:end-l.spline_degree], ε, l.init_grid; k=l.spline_degree, scale=device(half_quant.(l.init_τ)), basis_function=l.spline_function))
    end

    if (l.spline_string == "Cheby" || l.spline_string == "Gottlieb")
        spline_degree = l.spline_string == "Gottlieb" ? l.spline_degree + 1 : l.spline_degree
        return (coef=glorot_normal(rng, full_quant, l.in_dim, l.out_dim, spline_degree+1) .* (1 / (l.in_dim * (spline_degree + 1))), basis_τ=l.init_τ)
    else
        return l.τ_trainable ? (w_base=w_base, w_sp=w_sp, coef=coef, basis_τ=l.init_τ) : (w_base=w_base, w_sp=w_sp, coef=coef)
    end
end

function Lux.initialstates(rng::AbstractRNG, l::univariate_function)
    mask = ones(half_quant, l.in_dim, l.out_dim)
    return l.τ_trainable ? (mask=mask, grid=l.init_grid) : (mask=mask, grid=l.init_grid, basis_τ=half_quant.(l.init_τ))
end

function fwd(l, ps, st, x::AbstractArray{T}) where {T<:half_quant}
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

    coef, mask = ps.coef, st.mask
    τ = l.τ_trainable ? ps.basis_τ : st.basis_τ

    base = l.base_activation(x)
    y = coef2curve(x, st.grid, coef; k=l.spline_degree, scale=τ, basis_function=l.spline_function)
    
    if l.spline_string == "Cheby" || l.spline_string == "Gottlieb"
        return @tullio out[i, o, b] := y[i, o, b] * mask[i, o]
    else
        w_base, w_sp = ps.w_base, ps.w_sp
        return @tullio out[i, o, b] := (w_base[i, o] * base[i, b] + w_sp[i, o] * y[i, o, b]) * mask[i, o]
    end
end

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

end