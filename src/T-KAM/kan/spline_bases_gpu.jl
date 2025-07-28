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

using CUDA, Lux
using LinearAlgebra, NNlib

using ..Utils


function SplineMUL(
    """Top-level function for KAN with spline basis functions."""
    l::univariate_function{T,full_quant},
    ps::ComponentArray{T},
    x::AbstractArray{T},
    y::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
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
end

function (b::B_spline_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
)::AbstractArray{T} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    
    # 0-th degree
    if degree == 0
        grid_1 = grid[:, 1:(end-1)]
        grid_2 = grid[:, 2:end]

        # B0 is piecewise constant
        term1 = reshape(x, I, 1, S) .>= reshape(grid_1, I, G-1, 1)
        term2 = reshape(x, I, 1, S) .< reshape(grid_2, I, G-1, 1)
        B = T.(term1 .* term2)

    # k-th degree
    else   
        k = degree
        B = B_spline_basis(x, grid; degree = k-1)
        x = reshape(x, I, 1, S)

        numer1 = x .- grid[:, 1:(end-k-1)]
        denom1 = grid[:, (k+1):(end-1)] .- grid[:, 1:(end-k-1)]
        numer2 = grid[:, (k+2):end] .- x
        denom2 = grid[:, (k+2):end] .- grid[:, 2:(end-k)]
        B_i1 = B[:, 1:(end-1), :]
        B_i2 = B[:, 2:end, :]
        B = (numer1 ./ denom1) .* B_i1 .+ (numer2 ./ denom2) .* B_i2
    end

    return B
end

function (b::RBF_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    diff = reshape(x, I, 1, S) .- reshape(grid, I, G, 1)
    return exp.(-T(0.5) * (diff ./ (b.scale * σ)) .^ 2)
end

function (b::RSWAF_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
)::AbstractArray{T} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    diff = reshape(x, I, 1, S) .- reshape(grid, I, G, 1)
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
    x = repeat(reshape(x, size(x)..., 1), 1, 1, degree+1)
    linspace = collect(T, 0:degree) |> device
    return cos.(linspace' .* acos.(permutedims(x, [1, 3, 2])))
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
    ε::U = full_quant(1.0f-4),
)::AbstractArray{U} where {T<:half_quant,U<:full_quant}
    """Least sqaures fit of coefs from spline curves, (only for spline-types)."""
    J, S, O = size(x)..., size(y, 2)

    B = b(x, grid, σ) .|> full_quant
    G = size(B, 2)

    B = permutedims(B, [1, 3, 2]) # in_dim x b_size x n_grid

    coef = Array{U}(undef, J, O, G) |> pu
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

## Specific implementation for FFT basis functions ###
struct FFT_basis <: Lux.AbstractLuxLayer end

function (b::FFT_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::Tuple{AbstractArray{T},AbstractArray{T}} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    freq = reshape(x, I, 1, S) .* reshape(grid, I, G, 1)
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
    even_coef, odd_coef = reshape(coef[1, :, :, :], I, G, O, 1), reshape(coef[2, :, :, :], I, G, O, 1)
    return dropdims(sum(even .* even_coef .+ odd .* odd_coef, dims = 2), dims = 2)
end

end