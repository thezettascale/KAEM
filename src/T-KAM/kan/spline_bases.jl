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

using CUDA, ParallelStencil, Lux, ComponentArrays
using LinearAlgebra, NNlib

using ..Utils

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, half_quant, 3)
else
    @init_parallel_stencil(Threads, half_quant, 3)
end

@parallel_indices (i, o, b) function spl_kernel!(
    y::AbstractArray{T},
    x::AbstractArray{T},
    w_base::AbstractArray{T},
    w_sp::AbstractArray{T},
) where {T<:half_quant}
    y[i, o, b] = w_base[i, o] * x[i, b] + w_sp[i, o] * y[i, o, b]
    return nothing
end

function SplineMUL(
    l::Lux.AbstractLuxLayer,
    ps::ComponentArray{T},
    x::AbstractArray{T},
    y::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    """Top-level function for KAN with spline basis functions."""
    I, O, B = size(y)
    base = l.base_activation(x)
    @parallel (1:I, 1:O, 1:B) spl_kernel!(y, base, ps.w_base, ps.w_sp)
    return y
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

@parallel_indices (i, g, s) function B_spline_deg0!(
    B::AbstractArray{T},
    x::AbstractArray{T},
    grid::AbstractArray{T},
)::Nothing where {T<:half_quant}
    B[i, g, s, 1] = x[i, s] >= grid[i, g] && x[i, s] < grid[i, g+1] |> hq
    return nothing
end

@parallel_indices (i, g, s) function B_spline_degk!(
    B::AbstractArray{T},
    x::AbstractArray{T},
    grid::AbstractArray{T},
    k::Int,
)::Nothing where {T<:half_quant}
    xi = x[i, s]
    t_g = grid[i, g]
    t_gp = grid[i, g+1]
    t_gk = grid[i, g+k]
    t_gp1k = grid[i, g+k+1]

    B1 = B[i, g, s, k]
    B2 = B[i, g+1, s, k]

    denom1 = t_gk - t_g
    denom2 = t_gp1k - t_gp

    term1 = denom1 != 0 ? (xi - t_g) / denom1 * B1 : zero(T)
    term2 = denom2 != 0 ? (t_gp1k - xi) / denom2 * B2 : zero(T)

    B[i, g, s, k+1] = term1 + term2
    return nothing
end

function (b::B_spline_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
)::AbstractArray{T} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    B = @zeros(I, G-1, S, b.degree+1)
    @parallel (1:I, 1:(G-1), 1:S) B_spline_deg0!(B, x, grid)

    for k = 1:b.degree
        gmax = G - k - 1
        @parallel (1:I, 1:gmax, 1:S) B_spline_degk!(B, x, grid, k)
    end

    return B[:, 1:(G-b.degree-1), :, b.degree+1]
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

function (b::RBF_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    B = @zeros(I, G, S)
    @parallel (1:I, 1:G, 1:S) RBF_kernel!(B, x, grid, σ, b.scale)
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

function (b::RSWAF_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T};
)::AbstractArray{T} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    B = @zeros(I, G, S)
    @parallel (1:I, 1:G, 1:S) RSWAF_kernel!(B, x, grid, σ)
    return B
end

@parallel_indices (i, d, s) function Cheby_kernel!(
    B::AbstractArray{T},
    x::AbstractArray{T},
    σ::AbstractArray{T},
)::Nothing where {T<:half_quant}
    z = NNlib.tanh_fast(x[i, s] / σ[1])
    B[i, d, s] = cos((d-1) * acos(z))
    return nothing
end

function (b::Cheby_basis)(
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    B = @zeros(I, b.degree+1, S)
    @parallel (1:I, 1:(b.degree+1), 1:S) Cheby_kernel!(B, x, σ)
    return B
end

@parallel_indices (i, o, s) function spline_mul!(
    y::AbstractArray{T},
    spl::AbstractArray{T},
    coef::AbstractArray{T},
) where {T<:half_quant}
    acc = zero(T)
    @inbounds for g = 1:size(spl, 2)
        acc = acc + spl[i, g, s] * coef[i, o, g]
    end
    y[i, o, s] = acc
    return nothing
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
    y = @zeros(I, O, S)
    spl = b(x_eval, grid, σ)
    @parallel (1:I, 1:O, 1:S) spline_mul!(y, spl, coef)
    return y
end

function curve2coef(
    b::Lux.AbstractLuxLayer,
    x::AbstractArray{T},
    y::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    """Least sqaures fit of coefs from spline curves, (only for spline-types)."""
    J, S, O = size(x)..., size(y, 2)

    B = b(x, grid, σ) .|> full_quant
    y = y .|> full_quant
    G = size(B, 2)

    B = permutedims(B, [1, 3, 2]) # in_dim x b_size x n_grid

    coef = Array{full_quant}(undef, J, O, G) |> pu
    for i = 1:J
        for o = 1:O
            coef[i, o, :] .= B[i, :, :] \ y[i, o, :]
        end
    end

    replace!(coef, NaN => zero(full_quant))
    return T.(coef)
end

### Specific implementation for FFT basis functions ###
struct FFT_basis <: Lux.AbstractLuxLayer end

@parallel_indices (i, g, s) function FFT_kernel!(
    even::AbstractArray{T},
    odd::AbstractArray{T},
    x::AbstractArray{T},
    grid::AbstractArray{T},
    σ::AbstractArray{T},
)::Nothing where {T<:half_quant}
    freq = x[i, s] * grid[i, g]
    freq = 2π * freq * σ[1]
    even[i, g, s] = cos(freq)
    odd[i, g, s] = sin(freq)
    return nothing
end

function (b::FFT_basis)(
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
    @inbounds for g = 1:size(even, 2)
        acc = acc + even[i, g, s] * even_coef[i, o, g] + odd[i, g, s] * odd_coef[i, o, g]
    end
    y[i, o, s] = acc
    return nothing
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
    y = @zeros(I, O, S)
    @parallel (1:I, 1:O, 1:S) FFT_mul!(y, even, odd, coef[1, :, :, :], coef[2, :, :, :])
    return y
end
end
