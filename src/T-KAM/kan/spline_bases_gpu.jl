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
    SplineMUL

using CUDA, Lux, ComponentArrays
using LinearAlgebra, NNlib

using ..Utils


function SplineMUL(
    l::Lux.AbstractLuxLayer,
    ps::ComponentArray{T},
    x::AbstractArray{T},
    y::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    """Top-level function for KAN with spline basis functions."""
    I, O, B = size(y)
    base = l.base_activation(x)
    return ps.w_base .* reshape(base, I, 1, B) .+ ps.w_sp .* y
end

## Basis functions with Stencil loops ##
function extend_grid(grid::AbstractArray{T}; k_extend::Int = 0) where {T<:half_quant}
    h = (grid[:, end] - grid[:, 1]) / (size(grid, 2) - 1)

    for i = 1:k_extend
        grid = hcat(grid[:, 1:1] .- h, grid)
        grid = hcat(grid, grid[:, end:end] .+ h)
    end

    return grid
end

struct B_spline_basis <: Lux.AbstractLuxLayer
    degree::Int
end

struct RBF_basis <: Lux.AbstractLuxLayer
    scale::half_quant
end

struct RSWAF_basis <: Lux.AbstractLuxLayer end

struct Cheby_basis <: Lux.AbstractLuxLayer
    degree::Int
    lin::AbstractArray{half_quant}
end

function Cheby_basis(degree::Int)
    lin = collect(half_quant, 0:degree) |> pu
    return Cheby_basis(degree, lin)
end

function (b::B_spline_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
)::AbstractArray{T} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)

    # Initialize degree 0, piecewise const
    grid_1 = grid[:, 1:(end-1)]
    grid_2 = grid[:, 2:end]
    term1 = reshape(x, I, 1, S) .>= grid_1
    term2 = reshape(x, I, 1, S) .< grid_2
    B = T.(term1 .* term2)

    # Iteratively build up to degree k
    for d = 1:b.degree
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
        mask1 = T.(denom1 .!= 0)
        mask2 = T.(denom2 .!= 0)
        term1 = ((numer1 ./ denom1) .* B1) .* mask1
        term2 = ((numer2 ./ denom2) .* B2) .* mask2
        B = term1 + term2
    end

    return B
end

function (b::RBF_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    diff = reshape(x, I, 1, S) .- grid
    return exp.(-(diff ./ (b.scale .* σ) ./ 2) .^ 2)
end

function (b::RSWAF_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
)::AbstractArray{T} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    diff = reshape(x, I, 1, S) .- grid
    diff = NNlib.tanh_fast(diff ./ σ)
    return 1 .- diff .^ 2
end

function (b::Cheby_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    x = NNlib.tanh_fast(x) ./ σ
    x = repeat(reshape(x, size(x)..., 1), 1, 1, b.degree+1)
    return cos.(b.lin' .* acos.(permutedims(x, [1, 3, 2])))
end

function coef2curve_Spline(
    b::Lux.AbstractLuxLayer,
    x_eval::AbstractArray{T},
    grid::AbstractArray{T},
    coef::AbstractArray{T},
    σ::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    """Top-level function for coef multiplication for all splines."""
    I, S, O, G = size(x_eval)..., size(coef)[2:3]...
    G = b == Cheby_basis ? b.degree : G
    spl = reshape(b(x_eval, grid, σ), I, G, 1, S)
    coef = reshape(coef, I, G, O, 1)
    return dropdims(sum(spl .* coef, dims = 2), dims = 2)
end

function curve2coef(
    b::Lux.AbstractLuxLayer,
    x::AbstractArray{T},
    y::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::AbstractArray{U} where {T<:half_quant,U<:full_quant}
    """Least sqaures fit of coefs from spline curves, (only for spline-types)."""
    J, S, O = size(x)..., size(y, 2)

    B = b(x, grid, σ) .|> full_quant
    G = size(B, 2)

    B = permutedims(B, [1, 3, 2]) # in_dim x b_size x n_grid
    any(isnan.(B)) && error("NaN in B before least squares")

    coef = Array{U}(undef, J, O, G) |> pu
    for i = 1:J
        for o = 1:O
            coef[i, o, :] .= B[i, :, :] \ y[i, o, :]
        end
    end

    any(isnan.(coef)) && error("NaN in coef")
    return coef
end

## Specific implementation for FFT basis functions ###
struct FFT_basis <: Lux.AbstractLuxLayer end

function (b::FFT_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::Tuple{AbstractArray{T},AbstractArray{T}} where {T<:half_quant}
    I, S = size(x)
    freq = reshape(x, I, 1, 1, S) .* grid
    freq = T(2π) .* freq .* σ
    return cos.(freq), sin.(freq)
end

function coef2curve_FFT(
    b::Lux.AbstractLuxLayer,
    x_eval::AbstractArray{T},
    grid::AbstractArray{T},
    coef::AbstractArray{T},
    σ::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    I, S, O, G = size(x_eval)..., size(coef)[3:4]...
    even, odd = b(x_eval, grid, σ)
    even_coef, odd_coef =
        reshape(coef[1, :, :, :], I, G, O, 1), reshape(coef[2, :, :, :], I, G, O, 1)
    return dropdims(sum(even .* even_coef .+ odd .* odd_coef, dims = 2), dims = 2)
end

end
