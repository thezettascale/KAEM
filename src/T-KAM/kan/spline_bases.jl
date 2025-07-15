module spline_functions

export extend_grid,
    coef2curve_FFT,
    coef2curve_Spline,
    curve2coef,
    B_spline_basis,
    RBF_basis,
    RSWAF_basis,
    FFT_basis,
    Cheby_basis

using CUDA, KernelAbstractions, Tullio
using LinearAlgebra, NNlib

include("../../utils.jl")
using .Utils: removeNaN, device, half_quant, full_quant

function extend_grid(grid::AbstractArray{T}; k_extend::Int = 0) where {T<:half_quant}
    h = (grid[:, end] - grid[:, 1]) / (size(grid, 2) - 1)

    for i = 1:k_extend
        grid = hcat(grid[:, 1:1] .- h, grid)
        grid = hcat(grid, grid[:, end:end] .+ h)
    end

    return grid
end

function B_spline_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
    degree::Int = 3,
) where {T<:half_quant}
    I, S, G = size(x, 1), size(x, 2), size(grid, 2)

    # Initialize degree 0, piecewise const
    grid_1 = grid[:, 1:(end-1)]
    grid_2 = grid[:, 2:end]
    term1 = reshape(x, I, 1, S) .>= grid_1
    term2 = reshape(x, I, 1, S) .< grid_2
    B = T.(term1 .* term2)

    # Iteratively build up to degree k
    for d = 1:degree
        gmax = G - d - 1
        B1 = B[:, 1:gmax, :]
        B2 = B[:, 2:(gmax+1), :]
        grid_1 = grid[:, 1:gmax]
        grid_2 = grid[:, 2:(gmax+1)]
        grid_3 = grid[:, (d+1):(d+gmax)]
        grid_4 = grid[:, (d+2):(d+gmax+1)]

        numer1 = reshape(x, I, 1, S) .- grid_1
        denom1 = grid_3 .- grid_1
        numer2 = grid_4 .- reshape(x, I, 1, S)
        denom2 = grid_4 .- grid_2

        mask1 = T.(denom1 .!= 0) |> device
        mask2 = T.(denom2 .!= 0) |> device
        term1 = ((numer1 ./ denom1) .* B1) .* mask1
        term2 = ((numer2 ./ denom2) .* B2) .* mask2

        B = term1 + term2
    end

    return B
end

function RBF_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
    degree::Int = 3,
) where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    σ = ((maximum(grid) - minimum(grid)) / (size(grid, 2) - 1)) .* σ
    diff = reshape(x, I, 1, S) .- reshape(grid, I, G, 1)
    return exp.(-T(0.5) .* (diff ./ σ) .^ 2)
end

function RSWAF_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
    degree::Int = 3,
) where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    diff = reshape(x, I, 1, S) .- reshape(grid, I, G, 1)
    diff = NNlib.tanh_fast(diff ./ σ)
    return 1 .- diff .^ 2
end

function FFT_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
    degree::Int = 3,
) where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    freq = reshape(x, I, 1, S) .* grid
    freq = T(2π) .* freq .* σ
    return cos.(freq), sin.(freq)
end

function Cheby_basis(
    x::AbstractArray{T},
    grid_::AbstractArray{T},
    σ::AbstractArray{T};
    degree::Int = 3,
) where {T<:half_quant}
    x = NNlib.tanh_fast(x) ./ σ
    x = repeat(reshape(x, size(x)..., 1), 1, 1, degree+1)
    linspace = collect(0:degree) .|> T |> device
    return cos.(linspace' .* acos.(permutedims(x, [1, 3, 2])))
end

function coef2curve_FFT(
    x_eval::AbstractArray{T},
    grid::AbstractArray{T},
    coef::AbstractArray{T},
    σ::AbstractArray{T};
    k::Int = 3,
    basis_function::Function = FFT_basis,
) where {T<:half_quant}
    spl = basis_function(x_eval, grid, σ; degree = k)
    I, S, O, G = size(x_eval)..., size(coef)[2:3]...
    even, odd = spl
    even_coef, odd_coef =
        reshape(coef[1, :, :, :], I, G, O, 1), reshape(coef[2, :, :, :], I, G, O, 1)
    return dropdims(sum(even .* even_coef .+ odd .* odd_coef, dims = 2), dims = 2)
end

function coef2curve_Spline(
    x_eval::AbstractArray{T},
    grid::AbstractArray{T},
    coef::AbstractArray{T},
    σ::AbstractArray{T};
    k::Int = 3,
    basis_function::Function = FFT_basis,
) where {T<:half_quant}
    spl = basis_function(x_eval, grid, σ; degree = k)
    spl = reshape(spl, I, G, 1, S)
    coef = reshape(coef, I, G, O, 1)
    return dropdims(sum(spl .* coef, dims = 2), dims = 2)
end

function curve2coef(
    x::AbstractArray{T},
    y::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
    k::Int = 3,
    ε::U = full_quant(1.0f-4),
    basis_function::Function = B_spline_basis,
) where {T<:half_quant,U<:full_quant}
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
    J, S, O = size(x)..., size(y, 2)

    B = basis_function(x, grid, σ; degree = k) .|> full_quant
    G = size(B, 2)

    B = permutedims(B, [1, 3, 2]) # in_dim x b_size x n_grid

    coef = Array{U}(undef, J, O, G) |> device
    for i = 1:J
        for o = 1:O
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
