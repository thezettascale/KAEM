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
using Tullio, KernelAbstractions

using ..Utils


function SplineMUL(
    l::Lux.AbstractLuxLayer,
    ps::ComponentArray{T},
    x::AbstractArray{T},
    y::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    """Top-level function for KAN with spline basis functions."""
    x = l.base_activation(x)
    w_base, w_sp = ps.w_base, ps.w_sp
    return @tullio out[i, o, s] := w_base[i, o] * x[i, s] + w_sp[i, o] * y[i, o, s]
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
    @tullio term1[i, g, s] := x[i, s] >= grid_1[i, g]
    @tullio term2[i, g, s] := x[i, s] < grid_2[i, g]
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

        @tullio numer1[i, g, s] := x[i, s] - grid_1[i, g]
        @tullio denom1[i, g] := grid_3[i, g] - grid_1[i, g]
        @tullio numer2[i, g, s] := grid_4[i, g] - x[i, s]
        @tullio denom2[i, g] := grid_4[i, g] - grid_2[i, g]
        @tullio mask1[i, g] := denom1[i, g] != 0
        @tullio mask2[i, g] := denom2[i, g] != 0
        mask1 = T.(mask1)
        mask2 = T.(mask2)

        # Re-allocate memory for B since G (size) is changing
        @tullio B[i, g, s] :=
            (numer1[i, g, s] / denom1[i, g]) * B1[i, g, s] * mask1[i, g] +
            (numer2[i, g, s] / denom2[i, g]) * B2[i, g, s] * mask2[i, g]
    end

    return B
end

function (b::RBF_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    @tullio B[i, g, s] := exp(-((x[i, s] - grid[i, g]) / (b.scale * σ[d]))^2 / 2)
    return B
end

function (b::RSWAF_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
)::AbstractArray{T} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    @tullio B[i, g, s] := 1 - tanh((x[i, s] - grid[i, g]) / σ[d])^2
    return B
end

function (b::Cheby_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    @tullio B[i, g, s] := cos(b.lin[g] * acos(tanh(x[i, s] / σ[d])))
    return B
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
    spl = b(x_eval, grid, σ)
    @tullio y[i, o, s] := spl[i, g, s] * coef[i, o, g]
    return y
end

function curve2coef(
    b::Lux.AbstractLuxLayer,
    x::AbstractArray{T},
    y::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
    ε::full_quant = full_quant(1.0f-4),
)::AbstractArray{T} where {T<:half_quant}
    """Least sqaures fit of coefs from spline curves, (only for spline-types)."""
    J, S, O = size(x)..., size(y, 2)

    B = b(x, grid, σ) .|> full_quant
    y = y .|> full_quant
    G = size(B, 2)

    B = permutedims(B, [1, 3, 2]) # in_dim x b_size x n_grid

    eps = ε * I(G) |> pu
    coef = Array{full_quant}(undef, J, O, G) |> pu
    for i = 1:J
        for o = 1:O
            coef[i, o, :] .= (
                (B[i, :, :]' * B[i, :, :] + eps) # BtB
                \ (B[i, :, :]' * y[i, o, :]) # Bty
            )
        end
    end

    coef = ifelse.(isnan.(coef), zero(full_quant), coef) |> pu
    return T.(coef)
end

## Specific implementation for FFT basis functions ###
struct FFT_basis <: Lux.AbstractLuxLayer end

function (b::FFT_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::Tuple{AbstractArray{T},AbstractArray{T}} where {T<:half_quant}
    I, S = size(x)
    σ = T(2π) .* σ
    @tullio freq[i, g, s] := x[i, s] * grid[i, g] * σ[d]
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
    even_coef, odd_coef = coef[1, :, :, :], coef[2, :, :, :]
    @tullio y[i, o, s] := even[i, g, s] * even_coef[i, o, g] + odd[i, g, s] * odd_coef[i, o, g]
    return y
end

end
