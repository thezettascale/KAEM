module spline_functions

export extend_grid,
    coef2curve,
    curve2coef,
    B_spline_basis,
    RBF_basis,
    RSWAF_basis,
    FFT_basis,
    Cheby_basis,
    Gottlieb_basis

using CUDA, KernelAbstractions, Tullio
using LinearAlgebra, NNlib

include("../../utils.jl")
using .Utils: removeNaN, device, half_quant, full_quant

method = get(ENV, "method", "B-spline")

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
    grid::AbstractArray{T};
    degree::Int = 3,
    σ::Union{T,AbstractArray{T}} = one(half_quant),
) where {T<:half_quant}
    I, S, G = size(x, 1), size(x, 2), size(grid, 2)

    # Initialize degree 0, piecewise const
    grid_1 = grid[:, 1:(end-1)]
    grid_2 = grid[:, 2:end]
    @tullio term1[i, g, b] := x[i, b] >= grid_1[i, g]
    @tullio term2[i, g, b] := x[i, b] < grid_2[i, g]
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

        @tullio numer1[i, g, b] := x[i, b] - grid_1[i, g]
        @tullio denom1[i, g] := grid_3[i, g] - grid_1[i, g]
        @tullio numer2[i, g, b] := grid_4[i, g] - x[i, b]
        @tullio denom2[i, g] := grid_4[i, g] - grid_2[i, g]

        mask1 = denom1 .!= 0 |> device
        mask2 = denom2 .!= 0 |> device
        term1 = ((numer1 ./ denom1) .* B1) .* mask1
        term2 = ((numer2 ./ denom2) .* B2) .* mask2

        B = term1 + term2
    end

    return B
end

function RBF_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T};
    degree::Int = 3,
    σ::Union{T,AbstractArray{T}} = one(half_quant),
) where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    σ = ((maximum(grid) - minimum(grid)) / (size(grid, 2) - 1)) * σ
    @tullio diff[i, g, b] := x[i, b] - grid[i, g]
    return exp.(-T(0.5) * (diff ./ σ) .^ 2)
end

function RSWAF_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T};
    degree::Int = 3,
    σ::Union{T,AbstractArray{T}} = one(half_quant),
) where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    @tullio diff[i, g, b] := x[i, b] - grid[i, g]
    diff = NNlib.tanh_fast(diff ./ σ)
    return 1 .- diff .^ 2
end

function FFT_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T};
    degree::Int = 3,
    σ::Union{T,AbstractArray{T}} = one(half_quant),
) where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    @tullio freq[i, g, b] := x[i, b] * grid[i, g]
    freq = T(2π) .* freq .* σ
    return cos.(freq), sin.(freq)
end

function Cheby_basis(
    x::AbstractArray{T},
    grid_::AbstractArray{T};
    degree::Int = 3,
    σ::Union{T,AbstractArray{T}} = half_quant(1.1), # IMPORTANT; to make sure acos is well-defined, set σ > 1, e.g. 1.1
) where {T<:half_quant}
    x = NNlib.tanh_fast(x) ./ σ
    x = repeat(reshape(x, size(x)..., 1), 1, 1, degree+1)
    linspace = collect(0:degree) .|> T |> device
    return @tullio out[i, l, b] := cos(linspace[l] * acos(x[i, b, l]))
end

function Gottlieb_basis(
    x::AbstractArray{T},
    grid_::AbstractArray{T};
    degree::Int = 3,
    σ::Union{T,AbstractArray{T}} = one(half_quant),
) where {T<:half_quant}
    x = NNlib.sigmoid_fast(x)
    x = reshape(x, size(x)..., 1)
    B = ones(T, size(x, 1), size(x, 2), 1) |> device
    B = cat(B, 2σ .* x, dims = 3)
    for i = 3:(degree+1)
        y = 2(σ .+ (i-2)) .* x .* B[:, :, i-1] .- (σ .+ (2i-4)) .* B[:, :, i-2]
        B = cat(B, y, dims = 3)
    end
    return permutedims(B, [1, 3, 2])
end

function coef2curve(
    x_eval::AbstractArray{T},
    grid::AbstractArray{T},
    coef::AbstractArray{T};
    k::Int = 3,
    scale::Union{T,AbstractArray{T}} = one(half_quant),
    basis_function::Function = B_spline_basis,
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
    spl =
        isnothing(basis_function) ? B_spline_basis(x_eval, grid; degree = k) :
        basis_function(x_eval, grid; degree = k, σ = scale)

    !isa(spl, Tuple) && return @tullio y_eval[i, o, b] := spl[i, g, b] * coef[i, o, g]

    even, odd = spl
    even_coef, odd_coef = coef[1, :, :, :], coef[2, :, :, :]
    return @tullio y_eval[i, o, b] :=
        (even[i, g, b] * even_coef[i, o, g]) + (odd[i, g, b] * odd_coef[i, o, g])
end

function curve2coef(
    x::AbstractArray{T},
    y::AbstractArray{T},
    grid::AbstractArray{T};
    k::Int = 3,
    scale::Union{T,AbstractArray{T}} = one(half_quant),
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

    B = basis_function(x, grid; degree = k, σ = scale) .|> full_quant
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
