module spline_functions

export extend_grid,
    coef2curve_FFT,
    coef2curve_Spline,
    curve2coef,
    B_spline_basis,
    RBF_basis,
    RSWAF_basis,
    FFT_basis,
    Cheby_basis,
    ParallelStencil

using CUDA, KernelAbstractions, Tullio
using LinearAlgebra, NNlib

include("../../utils.jl")
using .Utils: removeNaN, device, half_quant, full_quant

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, half_quant, 3)
else
    @init_parallel_stencil(Threads, half_quant, 3)
end

function extend_grid(grid::AbstractArray{T}; k_extend::Int = 0) where {T<:half_quant}
    h = (grid[:, end] - grid[:, 1]) / (size(grid, 2) - 1)

    for i = 1:k_extend
        grid = hcat(grid[:, 1:1] .- h, grid)
        grid = hcat(grid, grid[:, end:end] .+ h)
    end

    return grid
end

@parallel_indices (i, g, s) function B_spline_deg0!(
    B::AbstractArray{T},
    x::AbstractArray{T},
    grid::AbstractArray{T},
)::Nothing where {T<:half_quant}
    B[i, g, s] = T(x[i, s] >= grid[g] && x[i, s] < grid[g+1])
    return nothing
end

@parallel_indices (i, g, s) function B_spline_degk!(
    B::AbstractArray{T},
    x::AbstractArray{T},
    grid::AbstractArray{T},
    d::Int,
)::Nothing where {T<:half_quant}
    B1 = B[i, s, g]
    B2 = B[i, s, g+1]
    grid_1 = grid[s, g]
    grid_2 = grid[s, g+1]
    grid_3 = grid[s, d+g]
    grid_4 = grid[s, d+g+1]

    numer1 = x[i, s] - grid_1
    denom1 = grid_3 - grid_1
    numer2 = grid_4 - x[i, s]
    denom2 = grid_4 - grid_2

    mask1 = T(denom1 != 0)
    mask2 = T(denom2 != 0)
    term1 = ((numer1 / denom1) * B1) * mask1
    term2 = ((numer2 / denom2) * B2) * mask2
    B[i, s, g] = term1 + term2
    return nothing
end

function B_spline_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
    degree::Int = 3,
)::Nothing where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    B = @zeros(I, G-1, S)
    @parallel (1:I, 1:(G-1), 1:S) B_spline_deg0!(B, x, grid)

    for d = 1:degree
        gmax = G - d - 1
        @parallel (1:I, 1:gmax, 1:S) B_spline_degk!(B, x, grid, d)
    end
    return B
end

@parallel_indices (i, g, s) function RBF_kernel!(
    B::AbstractArray{T},
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
    scale::T,
)::Nothing where {T<:half_quant}
    diff = x[i, s] - grid[i, g]
    B[i, g, s] = exp(-(diff / scale * σ[1]) ^ 2 / 2)
    return nothing
end

function RBF_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
    degree::Int = 3,
)::Nothing where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    B = @zeros(I, G, S)
    scale = (maximum(grid) - minimum(grid)) / (size(grid, 2) - 1)
    @parallel (1:I, 1:G, 1:S) RBF_kernel!(B, x, grid, σ, scale)
    return B
end

@parallel_indices (i, g, s) function RSWAF_kernel!(
    B::AbstractArray{T},
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::Nothing where {T<:half_quant}
    diff = x[i, s] - grid[i, g]
    B[i, g, s] = 1 - tanh(diff / σ[1]) ^ 2
    return nothing
end

function RSWAF_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
    degree::Int = 3,
)::Nothing where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    B = @zeros(I, G, S)
    @parallel (1:I, 1:G, 1:S) RSWAF_kernel!(B, x, grid, σ)
    return B
end

@parallel_indices (i, d, s) function Cheby_kernel!(
    B::AbstractArray{T},
    x::AbstractArray{T},
    lin::AbstractArray{T},
    σ::AbstractArray{T},
)::Nothing where {T<:half_quant}
    z = NNlib.tanh_fast(x[i, s] / σ[1])
    B[i, d, s] = cos(lin[d] * acos(z))
    return nothing
end

function Cheby_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
    degree::Int = 3,
)::Nothing where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    B = @zeros(I, degree+1, S)
    lin = collect(T, 0:degree) |> device
    @parallel (1:I, 1:(degree+1), 1:S) Cheby_kernel!(B, x, lin, σ)
    return B
end

@parallel_indices (i, o, s) function spline_mul!(
    y::AbstractArray{T},
    spl::AbstractArray{T},
    coef::AbstractArray{T},
) where {T<:half_quant}
    acc = zero(T)
    for g = 1:size(spl, 2)
        acc += spl[i, g, s] * coef[i, o, g]
    end
    y[i, o, s] = acc
    return nothing
end

function coef2curve_Spline(
    x_eval::AbstractArray{T},
    grid::AbstractArray{T},
    coef::AbstractArray{T},
    σ::AbstractArray{T};
    k::Int = 3,
    basis_function::Function = RBF_basis,
)::AbstractArray{T} where {T<:half_quant}
    I, S, O, G = size(x_eval)..., size(coef)[2:3]...
    spl = @zeros(I, G, S)
    y = @zeros(I, O, S)

    spl = basis_function(x_eval, grid, σ; degree = k)
    @parallel (1:I, 1:O, 1:S) spline_mul!(y, spl, coef)
    return y
end

@parallel_indices (i, g, s) function FFT_kernel!(
    even::AbstractArray{T},
    odd::AbstractArray{T},
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::Nothing where {T<:half_quant}
    freq = x[i, s] * grid[i, g]
    freq = T(2π) * freq * σ[1]
    even[i, g, s] = cos(freq)
    odd[i, g, s] = sin(freq)
    return nothing
end

function FFT_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::Tuple{AbstractArray{T},AbstractArray{T}} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    even = @zeros(I, G, S)
    odd = @zeros(I, G, S)
    @parallel (1:I, 1:G, 1:S) FFT_kernel!(even, odd, x, grid, σ)
    return even, odd
end

@parallel_indices (i, o, s) function FFT_mul!(
    y::AbstractArray{T},
    even::AbstractArray{T},
    odd::AbstractArray{T},
    even_coef::AbstractArray{T},
    odd_coef::AbstractArray{T},
)::Nothing where {T<:half_quant}
    acc = zero(T)
    for g = 1:size(even, 2)
        acc += even[i, g, s] * even_coef[i, o, g] + odd[i, g, s] * odd_coef[i, o, g]
    end
    y[i, o, s] = acc
    return nothing
end

function coef2curve_FFT(
    x_eval::AbstractArray{T},
    grid::AbstractArray{T},
    coef::AbstractArray{T},
    σ::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    I, S, O, G = size(x_eval)..., size(coef)[2:3]...
    spl = FFT_basis(x_eval, grid, σ)
    @parallel (1:I, 1:O, 1:S) FFT_mul!(y, spl, coef)
    return y
end

function curve2coef(
    x::AbstractArray{T},
    y::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
    k::Int = 3,
    ε::U = full_quant(1.0f-4),
    basis_function::Function = B_spline_basis,
)::AbstractArray{U} where {T<:half_quant,U<:full_quant}
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
