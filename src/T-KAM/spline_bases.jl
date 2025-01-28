module spline_functions

export extend_grid, coef2curve, curve2coef, B_spline_basis, RBF_basis, RSWAF_basis

using CUDA, KernelAbstractions
using Tullio, LinearAlgebra
using NNlib

include("../utils.jl")
using .Utils: removeNaN, device, half_quant, full_quant

method = get(ENV, "method", "B-spline") 

function extend_grid(grid::AbstractArray{half_quant}; k_extend::Int64=0)
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
    x::AbstractArray{half_quant},
    grid::AbstractArray{half_quant};
    degree::Int64=3, 
    σ::Union{half_quant, AbstractArray{half_quant}}=half_quant(1)
    )
    """
    Compute the B-spline basis functions for a batch of points x and a grid of knots.

    Args:
        x: A matrix of size (b, i) containing the points at which to evaluate the B-spline basis functions.
        grid: A matrix of size (i, g) containing the grid of knots.
        degree: The degree of the B-spline basis functions.

    Returns:
        A matrix of size (b, i, g) containing the B-spline basis functions evaluated at the points x.
    """
    x = reshape(x, size(x)..., 1) 
    grid = reshape(grid, 1, size(grid)...) 
    
    # 0-th degree
    if degree == 0
        grid_1 = grid[:, :, 1:end-1] 
        grid_2 = grid[:, :, 2:end] 
    
        # B0 is piecewise constant
        @tullio term1[b, i, g] := x[b, i, 1] >= grid_1[1, i, g]
        @tullio term2[b, i, g] := x[b, i, 1] < grid_2[1, i, g]
        term1 = half_quant.(term1)
        term2 = half_quant.(term2)

        @tullio B[b, i, g] := term1[b, i, g] * term2[b, i, g]

    else
        # k-th degree
        k = degree
        B = B_spline_basis(x[:, :, 1], grid[1, :, :]; degree=k-1)
        

        numer1 = x .- grid[:, :, 1:(end - k - 1)]
        denom1 = grid[:, :, (k + 1):end-1] .- grid[:, :, 1:(end - k - 1)]
        numer2 = grid[:, :, (k + 2):end] .- x
        denom2 = grid[:, :, (k + 2):end] .- grid[:, :, 2:(end - k)]
        B_i1 = B[:, :, 1:end - 1]
        B_i2 = B[:, :, 2:end]

        @tullio B[b, i, g] := (numer1[b, i, g] / denom1[1, i, g]) * B_i1[b, i, g] + (numer2[b, i, g] / denom2[1, i, g]) * B_i2[b, i, g]
    end
    
    return B
end

function RBF_basis(
    x::AbstractArray{half_quant},
    grid::AbstractArray{half_quant};
    degree::Int64=3, 
    σ::Union{half_quant, AbstractArray{half_quant}}=half_quant(1)
    )
    """
    Compute the RBF basis functions for a batch of points x and a grid of knots.

    Args:
        x: A matrix of size (b, i) containing the points at which to evaluate the RBF basis functions.
        grid: A matrix of size (i, g) containing the grid of knots.
        σ: Tuning for the bandwidth (standard deviation) of the RBF kernel.

    Returns:
        A matrix of size (b, i, g) containing the RBF basis functions evaluated at the points x.
    """
    σ = ((maximum(grid) - minimum(grid)) / (size(grid, 2) - 1)) * σ
    @tullio diff[b, i, g] := x[b, i] - grid[i, g] 
    diff = diff ./ σ
    return @tullio B[b, i, g] := exp(-5f-1 * (diff[b, i, g])^2)    
end

function RSWAF_basis(
    x::AbstractArray{half_quant},
    grid::AbstractArray{half_quant};
    degree::Int64=3, 
    σ::Union{half_quant, AbstractArray{half_quant}}=half_quant(1)
    )
    """
    Compute the RSWAF basis functions for a batch of points x and a grid of knots.
        Be careful of vanishing gradients when using this in a deep network.

    Args:
        x: A matrix of size (b, i) containing the points at which to evaluate the RSWAF basis functions.
        grid: A matrix of size (i, g) containing the grid of knots.
        σ: Tuning for the bandwidth (standard deviation) of the RSWAF kernel.

    Returns:
        A matrix of size (b, i, g) containing the RSWAF basis functions evaluated at the points x.
    """
    # Fast tanh may cause stability problems, but is faster. If problematic, use base tanh instead. 
    @tullio diff[b, i, g] := x[b, i] - grid[i, g] 
    diff = NNlib.tanh_fast(diff ./ σ)     
    return @tullio B[b, i, g] := 1 - diff[b, i, g]^2
end

function coef2curve(
    x_eval::AbstractArray{half_quant},
    grid::AbstractArray{half_quant},
    coef::AbstractArray{half_quant};
    k::Int64=3, 
    scale::Union{half_quant, AbstractArray{half_quant}}=half_quant(1), 
    basis_function::Function=B_spline_basis
    )
    """
    Compute the B-spline curves from the B-spline coefficients.

    Args:
        x_eval: A matrix of size (b, i) containing the points at which to evaluate the B-spline curves.
        grid: A matrix of size (b, g) containing the grid of knots.
        coef: A matrix of size (i, o, g) containing the B-spline coefficients.
        k: The degree of the B-spline basis functions.

    Returns:
        A matrix of size (b, i, o) containing the B-spline curves evaluated at the points x_eval.
    """
    splines = isnothing(basis_function) ? B_spline_basis(x_eval, grid; degree=k) : basis_function(x_eval, grid; degree=k, σ=scale)
    return @tullio y_eval[b, i, o] := splines[b, i, g] * coef[i, o, g]
end

function curve2coef(
    x_eval::AbstractArray{half_quant},
    y_eval::AbstractArray{half_quant},
    grid::AbstractArray{half_quant};
    k::Int64=3,
    scale::Union{half_quant, AbstractArray{half_quant}}=half_quant(1), 
    ε::half_quant=half_quant(1e-4), 
    basis_function::Function=B_spline_basis
    )
    """
    Convert B-spline curves to B-spline coefficients using least squares.
    This will not work for poly-KANs.

    Args:
        x_eval: A matrix of size (b, i) containing the points at which the B-spline curves were evaluated.
        y_eval: A matrix of size (b, i, o) containing the B-spline curves evaluated at the points x_eval.
        grid: A matrix of size (b, g) containing the grid of knots.
        k: The degree of the B-spline basis functions.

    Returns:
        A matrix of size (i, o, g) containing the B-spline coefficients.
    """
    b_size, in_dim = size(x_eval)
    out_dim = size(y_eval, 3)

    # b_size x in_dim x n_grid
    B = basis_function(x_eval, grid; degree=k, σ=scale)  
    n_grid = size(B, 3)

    B = reshape(B, in_dim, 1, b_size, n_grid)
    B = repeat(B, 1, out_dim, 1, 1) # in_dim x out_dim x b_size x n_grids

    y_eval = permutedims(y_eval, [2, 3, 1]) # in_dim x out_dim x b_size
    y_eval = reshape(y_eval, size(y_eval)..., 1)

    # Get BtB and Bty
    Bt = permutedims(B, [1, 2, 4, 3])
    
    @tullio BtB[i, o, g, j] := Bt[i, o, g, b] * B[i, o, b, j] # in_dim x out_dim x n_grids x n_grids
    n1, n2, n, _ = size(BtB)
    eye = Matrix{half_quant}(I, n, n) .* ε |> device
    eye = reshape(eye, 1, 1, n, n)
    eye = repeat(eye, n1, n2, 1, 1)
    BtB = BtB + eye 
    
    @tullio Bty[i, o, g, j] := Bt[i, o, g, b] * y_eval[i, o, b, j]

    BtB, Bty = BtB .|> full_quant, Bty .|> full_quant
    
    # x = (BtB)^-1 * Bty
    coef = zeros(full_quant, 0, out_dim, n_grid) |> device
    for i in 1:in_dim
        coef_ = zeros(full_quant, 0, n_grid) |> device
        for o in 1:out_dim
            lstq = qr(BtB[i, o, :, :]) \ Bty[i, o, :, :]
            lstq = lstq |> permutedims
            coef_ = vcat(coef_, lstq .|> full_quant)
        end
        coef_ = reshape(coef_, 1, size(coef_)...)
        coef = vcat(coef, coef_)
    end

    any(isnan.(coef)) && error("NaN in coef")
    return coef
end
end