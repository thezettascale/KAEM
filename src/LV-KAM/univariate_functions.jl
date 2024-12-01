module spline_functions

export extend_grid, B_spline_basis, coef2curve, curve2coef

using Tullio, LinearAlgebra
using NNlib

include("../utils.jl")
using .Utils: removeNaN, device

method = get(ENV, "method", "B-spline") 

function extend_grid(grid; k_extend=0)
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

function B_spline_basis(x, grid; degree::Int64, σ=nothing)
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
        term1 = @tullio res[i, j, l] := x[i, j, k] >= grid_1[p, j, l]
        term2 = @tullio res[i, j, l] := x[i, j, k] < grid_2[p, j, l]
        term1 = Float32.(term1)
        term2 = Float32.(term2)

        B = @tullio res[d, p, n] := term1[d, p, n] * term2[d, p, n]

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

        B = @tullio out[d, n, m] := (numer1[d, n, m] / denom1[1, n, m]) * B_i1[d, n, m] + (numer2[d, n, m] / denom2[1, n, m]) * B_i2[d, n, m]
    end
    
    # B = removeNaN(B)
    # any(isnan.(B)) && error("NaN in B") 
    any(isnan.(B)) && println("NaN in B-spline basis")
    
    return B
end

function RBF_basis(x, grid; degree=nothing, σ=1f0)
    """
    Compute the RBF basis functions for a batch of points x and a grid of knots.

    Args:
        x: A matrix of size (b, i) containing the points at which to evaluate the RBF basis functions.
        grid: A matrix of size (i, g) containing the grid of knots.
        σ: Tuning for the bandwidth (standard deviation) of the RBF kernel.

    Returns:
        A matrix of size (b, i, g) containing the RBF basis functions evaluated at the points x.
    """
    x = reshape(x, size(x)..., 1)
    grid = reshape(grid, 1, size(grid)...)

    σ = ((maximum(grid) - minimum(grid)) / (size(grid, 3) - 1)) * σ
    norm = Float32.(1 ./ 2π) ./ σ
    diff = @tullio res[d, n, m] := x[d, n, 1] - grid[1, n, m] 
    diff = diff ./ σ

    B = @tullio res[d, n, m] := exp(-5f-1 * (diff[d, n, m])^2)

    # any(isnan.(B)) && error("NaN in B")
    any(isnan.(B)) && println("NaN in RBF basis")
    
    return norm .* B
end

function RSWAF_basis(x, grid; degree=nothing, σ=10f0)
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
    
    x = reshape(x, size(x)..., 1)
    grid = reshape(grid, 1, size(grid)...)

    diff = @tullio res[d, n, m] := x[d, n, 1] - grid[1, n, m]
    
    # Fast tanh may cause stability problems, but I was more interested in speed. If problematic, use base tanh instead. 
    tan_diff = NNlib.tanh_fast(diff ./ σ) 
    # tan_diff = tanh.(diff ./ σ)
    
    B = @tullio res[d, n, m] := 1 - tan_diff[d, n, m]^2

    # any(isnan.(B)) && error("NaN in B")
    any(isnan.(B)) && println("NaN in RSWAF basis")

    return B
end

BasisFcn = Dict(
    "B-spline" => B_spline_basis,
    "RBF" => RBF_basis,
    "RSWAF" => RSWAF_basis,
)[method]

function coef2curve(x_eval, grid, coef; k::Int64, scale=1f0)
    """
    Compute the B-spline curves from the B-spline coefficients.

    Args:
        x_eval: A matrix of size (b, i) containing the points at which to evaluate the B-spline curves.
        grid: A matrix of size (b, g) containing the grid of knots.
        coef: A matrix of size (i, o, nc) containing the B-spline coefficients.
        k: The degree of the B-spline basis functions.

    Returns:
        A matrix of size (b, i, o) containing the B-spline curves evaluated at the points x_eval.
    """
    splines = BasisFcn(x_eval, grid; degree=k, σ=scale)
    y_eval = @tullio out[i, j, l] := splines[i, j, p] * coef[j, l, p]
    return y_eval
end

function curve2coef(x_eval, y_eval, grid; k::Int64, scale=1f0, ε=0f0)
    """
    Convert B-spline curves to B-spline coefficients using least squares.
    This will not work for poly-KANs.

    Args:
        x_eval: A matrix of size (b, i) containing the points at which the B-spline curves were evaluated.
        y_eval: A matrix of size (b, i, o) containing the B-spline curves evaluated at the points x_eval.
        grid: A matrix of size (b, g) containing the grid of knots.
        k: The degree of the B-spline basis functions.

    Returns:
        A matrix of size (i, o, nc) containing the B-spline coefficients.
    """
    b_size = size(x_eval, 1)
    in_dim = size(x_eval, 2)
    out_dim = size(y_eval, 3)

    B = BasisFcn(x_eval, grid; degree=k, σ=scale)  # b_size x in_dim x n_coeff

    n_coeff = size(B, 3)

    B = permutedims(B, [2, 1, 3])
    B = reshape(B, in_dim, 1, b_size, size(B, 3))
    B = repeat(B, 1, out_dim, 1, 1) # in_dim x out_dim x b_size x n_coeffs

    y_eval = permutedims(y_eval, [2, 3, 1]) # in_dim x out_dim x b_size
    y_eval = reshape(y_eval, size(y_eval)..., 1)

    # Get BtB and Bty
    Bt = permutedims(B, [1, 2, 4, 3])
    
    BtB = @tullio out[i, j, m, p] := Bt[i, j, m, n] * B[i, j, n, p] # in_dim x out_dim x n_coeffs x n_coeffs
    n1, n2, n, _ = size(BtB)
    eye = Matrix{Float32}(I, n, n) .* ε |> device
    eye = reshape(eye, 1, 1, n, n)
    eye = repeat(eye, n1, n2, 1, 1)
    BtB = BtB + eye 
    
    Bty = @tullio out[i, j, m, p] := Bt[i, j, m, n] * y_eval[i, j, n, p]
    
    # x = (BtB)^-1 * Bty
    coef = zeros(Float32, 0, out_dim, n_coeff) |> device
    for i in 1:in_dim
        coef_ = zeros(Float32, 0, n_coeff) |> device
        for j in 1:out_dim
            lstq = qr(BtB[i, j, :, :]) \ Bty[i, j, :, :]
            lstq = lstq |> permutedims
            coef_ = vcat(coef_, lstq)
        end
        coef_ = reshape(coef_, 1, size(coef_)...)
        coef = vcat(coef, coef_)
    end

    # any(isnan.(coef)) && error("NaN in coef")
    return coef
end
end