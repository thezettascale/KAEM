module spline_functions

export extend_grid, coef2curve, curve2coef, B_spline_basis, RBF_basis, RSWAF_basis, FFT_basis, Morlet_basis, Shannon_basis

using CUDA, KernelAbstractions
using Tullio, LinearAlgebra, NNlib

include("../utils.jl")
using .Utils: removeNaN, device, half_quant, full_quant

method = get(ENV, "method", "B-spline") 

function extend_grid(grid::AbstractArray{T}; k_extend::Int=0) where {T<:half_quant}
    """
    Extend the grid of knots to include boundary knots.

    Args:
        grid: A matrix of size (i, g) containing the grid of knots.
        k_extend: The number of boundary knots to add to the grid.
    
    Returns:
        A matrix of size (i, g + 2 * k_extend) containing the extended grid of knots.
    """
    h = (grid[:, end] - grid[:, 1]) / (size(grid, 2) - 1)

    for i in 1:k_extend
        grid = hcat(grid[:, 1:1] .- h, grid)
        grid = hcat(grid, grid[:, end:end] .+ h)
    end
    
    return grid
end

function B_spline_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T};
    degree::Int=3, 
    σ::Union{T, AbstractArray{T}}=one(half_quant)
    ) where {T<:half_quant}
    """
    Compute the B-spline basis functions for a batch of points x and a grid of knots.

    Args:
        x: A matrix of size (i, b) containing the points at which to evaluate the B-spline basis functions.
        grid: A matrix of size (i, g) containing the grid of knots.
        degree: The degree of the B-spline basis functions.

    Returns:
        A matrix of size (i, g, b) containing the B-spline basis functions evaluated at the points x.
    """

    in_size, sample_size, G = size(x)..., size(grid, 2)
    
    # 0-th degree
    if degree == 0
        grid_1 = grid[:, 1:end-1] 
        grid_2 = grid[:, 2:end] 
    
        # B0 is piecewise constant
        @tullio term1[i, g, b] := x[i, b] >= grid_1[i, g]
        @tullio term2[i, g, b] := x[i, b] < grid_2[i, g]
        term1 = T.(term1)
        term2 = T.(term2)

        B = term1 .* term2

    else
        # k-th degree
        k = degree
        B = B_spline_basis(x, grid; degree=k-1)
        
        x = reshape(x, in_size, 1, sample_size)

        numer1 = x .- grid[:, 1:(end - k - 1)]
        denom1 = grid[:, (k + 1):end-1] .- grid[:, 1:(end - k - 1)]
        numer2 = grid[:, (k + 2):end] .- x
        denom2 = grid[:, (k + 2):end] .- grid[:, 2:(end - k)]
        B_i1 = B[:, 1:end - 1, :]
        B_i2 = B[:, 2:end, :]

        B = (numer1 ./ denom1) .* B_i1 .+ (numer2 ./ denom2) .* B_i2
    end
    
    return B
end

function RBF_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T};
    degree::Int=3, 
    σ::Union{T, AbstractArray{T}}=one(half_quant)
    ) where {T<:half_quant}
    """
    Compute the RBF basis functions for a batch of points x and a grid of knots.

    Args:
        x: A matrix of size (i, b) containing the points at which to evaluate the RBF basis functions.
        grid: A matrix of size (i, g) containing the grid of knots.
        σ: Tuning for the bandwidth (standard deviation) of the RBF kernel.

    Returns:
        A matrix of size (i, g, b) containing the RBF basis functions evaluated at the points x.
    """
    σ = ((maximum(grid) - minimum(grid)) / (size(grid, 2) - 1)) * σ
    @tullio diff[i, g, b] := x[i, b] - grid[i, g] 
    return exp.(-T(0.5) * (diff ./ σ).^2)  
end

function RSWAF_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T};
    degree::Int=3, 
    σ::Union{T, AbstractArray{T}}=one(half_quant)
    ) where {T<:half_quant}
    """
    Compute the RSWAF basis functions for a batch of points x and a grid of knots.
        Be careful of vanishing gradients when using this in a deep network.

    Args:
        x: A matrix of size (i, b) containing the points at which to evaluate the RSWAF basis functions.
        grid: A matrix of size (i, g) containing the grid of knots.
        σ: Tuning for the bandwidth (standard deviation) of the RSWAF kernel.

    Returns:
        A matrix of size (i, g, b) containing the RSWAF basis functions evaluated at the points x.
    """
    # Fast tanh may cause stability problems, but is faster. If problematic, use base tanh instead. 
    @tullio diff[i, g, b] := x[i, b] - grid[i, g] 
    diff = NNlib.tanh_fast(diff ./ σ)     
    return 1 .- diff.^2
end

function FFT_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T};
    degree::Int=3, 
    σ::Union{T, AbstractArray{T}}=one(half_quant)
    ) where {T<:half_quant}
    """
    Compute the FFT basis functions for a batch of points x and a grid of knots.

    Args:
        x: A matrix of size (i, b) containing the points at which to evaluate the FFT basis functions.
        grid: A matrix of size (i, g) containing the grid of knots.
        σ: Tuning for the bandwidth (standard deviation) of the FFT kernel.

    Returns:
        A matrix of size (i, g, b) containing the FFT basis functions evaluated at the points x.
    """
    @tullio freq[i, g, b] := x[i, b] * grid[i, g]
    freq = T(2π) .* freq .* σ
    return cos.(freq), sin.(freq)
end

function coef2curve(
    x_eval::AbstractArray{T},
    grid::AbstractArray{T},
    coef::AbstractArray{T};
    k::Int=3, 
    scale::Union{T, AbstractArray{T}}=one(half_quant), 
    basis_function::Function=B_spline_basis
    ) where {T<:half_quant}
    """
    Compute the B-spline curves from the B-spline coefficients.

    Args:
        x_eval: A matrix of size (i, b) containing the points at which to evaluate the B-spline curves.
        grid: A matrix of size (g, b) containing the grid of knots.
        coef: A matrix of size (i, o, g) containing the B-spline coefficients.
        k: The degree of the B-spline basis functions.

    Returns:
        A matrix of size (i, o, b) containing the B-spline curves evaluated at the points x_eval.
    """

    splines = isnothing(basis_function) ? B_spline_basis(x_eval, grid; degree=k) : basis_function(x_eval, grid; degree=k, σ=scale)
    !isa(splines, Tuple) && return @tullio y_eval[i, o, b] := splines[i, g, b] * coef[i, o, g]

    even, odd = splines
    even_coef, odd_coef = coef[1, :, :, :], coef[2, :, :, :]
    return @tullio y_eval[i, o, b] := (even[i, g, b] * even_coef[i, o, g]) + (odd[i, g, b] * odd_coef[i, o, g])
end

function curve2coef(
    x::AbstractArray{T},
    y::AbstractArray{T},
    grid::AbstractArray{T};
    k::Int=3,
    scale::Union{T, AbstractArray{T}}=one(half_quant), 
    ε::U=full_quant(1f-4), 
    basis_function::Function=B_spline_basis
    ) where {T<:half_quant, U<:full_quant}
    """
    Convert B-spline curves to B-spline coefficients using least squares.
    This will not work for poly-KANs. CuSolver works best for higher precisions.

    Args:
        x: A matrix of size (i, b) containing the points at which the B-spline curves were evaluated.
        y: A matrix of size (i, o, b) containing the B-spline curves evaluated at the points x_eval.
        grid: A matrix of size (i, g) containing the grid of knots.
        k: The degree of the B-spline basis functions.

    Returns:
        A matrix of size (i, o, g) containing the B-spline coefficients.
    """
    in_size, sample_size, out_size = size(x)..., size(y, 2)

    B = basis_function(x, grid; degree=k, σ=scale) .|> full_quant
    G = size(B, 2)

    B = permutedims(B, [1, 3, 2]) # in_dim x b_size x n_grid

    coef = Array{U}(undef, in_size, out_size, G) |> device
    for i in 1:in_size
        for o in 1:out_size
            coef[i, o, :] .= (
                (B[i, :, :]' * B[i, :, :] + ε * I) # BtB
                \ (B[i, :, :]' * y[i, o, :]) # Bty
                ) 
        end
    end

    any(isnan.(coef)) && error("NaN in coef")
    return coef
end
end