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

using ComponentArrays, LinearAlgebra, Lux, Accessors
using Flux: tanh_fast

using ..Utils

include("lst_sq.jl")
using .LstSqSolver

function extend_grid(
        grid;
        k_extend = 0,
    )
    h = (grid[:, end] - grid[:, 1]) / (size(grid, 2) - 1)

    for i in 1:k_extend
        grid = hcat(grid[:, 1:1] .- h, grid)
        grid = hcat(grid, grid[:, end:end] .+ h)
    end

    return grid
end

struct B_spline_basis <: AbstractBasis
    degree
    I
    O
    G
    S
    k_mask
    lower_mask
    upper_mask
end

function B_spline_basis(degree::Int, I::Int, O::Int, G::Int, S::Int)
    k_mask = Lux.f32((1:G) .== (1:G)')
    lower_mask = Lux.f32((1:G) .> (1:G)')
    upper_mask = Lux.f32((1:G) .>= (1:G)')
    return B_spline_basis(
        degree,
        I,
        O,
        G,
        S,
        k_mask,
        lower_mask,
        upper_mask,
    )
end

struct RBF_basis <: AbstractBasis
    I
    O
    G
    S
    k_mask
    lower_mask
    upper_mask
end

function RBF_basis(I::Int, O::Int, G::Int, S::Int)
    k_mask = Lux.f32((1:G) .== (1:G)')
    lower_mask = Lux.f32((1:G) .> (1:G)')
    upper_mask = Lux.f32((1:G) .>= (1:G)')
    return RBF_basis(
        I,
        O,
        G,
        S,
        k_mask,
        lower_mask,
        upper_mask,
    )
end

struct RSWAF_basis <: AbstractBasis
    I
    O
    G
    S
    k_mask
    lower_mask
    upper_mask
end

function RSWAF_basis(I::Int, O::Int, G::Int, S::Int)
    k_mask = Lux.f32((1:G) .== (1:G)')
    lower_mask = Lux.f32((1:G) .> (1:G)')
    upper_mask = Lux.f32((1:G) .>= (1:G)')
    return RSWAF_basis(
        I,
        O,
        G,
        S,
        k_mask,
        lower_mask,
        upper_mask,
    )
end

struct Cheby_basis <: AbstractBasis
    degree
    lin
    I
    O
    G
    S
    k_mask
    lower_mask
    upper_mask
end

function Cheby_basis(degree::Int, I::Int, O::Int, S::Int)
    G = degree + 1
    lin = Lux.f32((0:degree)')
    k_mask = Lux.f32((1:G) .== (1:G)')
    lower_mask = Lux.f32((1:G) .> (1:G)')
    upper_mask = Lux.f32((1:G) .>= (1:G)')
    return Cheby_basis(
        degree,
        lin,
        I,
        O,
        G,
        S,
        k_mask,
        lower_mask,
        upper_mask,
    )
end

# Broken
function (b::B_spline_basis)(
        x,
        grid,
        σ,
        scale;
        init::Bool = false,
    )
    I, G, S = b.I, b.G - 1, b.S
    x = PermutedDimsArray(view(x, :, :, :), (1, 3, 2))

    # B0
    grid_1 = @view grid[:, 1:(end - 1)]
    grid_2 = @view grid[:, 2:end]
    B = Float32.((x .>= grid_1) .* (x .< grid_2))

    # Iteratively build up to degree k
    for d in 1:b.degree
        gmax = G - d - 1
        B1 = @view B[:, 1:gmax, :]
        B2 = @view B[:, 2:(gmax + 1), :]
        g1 = @view grid[:, 1:gmax, :]
        g2 = @view grid[:, 2:(gmax + 1), :]
        g3 = @view grid[:, (d + 1):(d + gmax), :]
        g4 = @view grid[:, (d + 2):(d + gmax + 1), :]

        denom1 = g3 .- g1
        denom2 = g4 .- g2

        mask1 = Float32.(denom1 .!= 0)
        mask2 = Float32.(denom2 .!= 0)

        numer1 = x .- g1
        numer2 = g4 .- x

        B = @. ((numer1 / denom1) * B1 * mask1 + (numer2 / denom2) * B2 * mask2)
    end

    return B
end

function (b::RBF_basis)(
        x,
        grid,
        σ,
        scale;
        init::Bool = false,
    )
    x_3d = PermutedDimsArray(view(x, :, :, :), (1, 3, 2))
    return exp.(-((x_3d .- grid) ./ (scale .* σ)) .^ 2 ./ 2)
end

function (b::RSWAF_basis)(
        x,
        grid,
        σ,
        scale;
        init::Bool = false,
    )
    x_3d = PermutedDimsArray(view(x, :, :, :), (1, 3, 2))
    diff = tanh_fast((x_3d .- grid) ./ σ)
    return 1.0f0 .- diff .^ 2
end

# Not working
function (b::Cheby_basis)(
        x,
        grid,
        σ,
        scale;
        init::Bool = false,
    )
    x_3d = PermutedDimsArray(view(x, :, :, :), (1, 3, 2))
    x_3d = (tanh_fast(x_3d) ./ σ)
    return cos.(acos.(x_3d) .* b.lin)
end

function coef2curve_Spline(
        b,
        x_eval,
        grid,
        coef,
        σ,
        scale;
        init::Bool = false,
    )
    spl = b(x_eval, grid, σ, scale)
    spl_4d = PermutedDimsArray(view(spl, :, :, :, :), (1, 4, 3, 2))
    coef_4d = PermutedDimsArray(view(coef, :, :, :, :), (1, 2, 4, 3))
    curve = spl_4d .* coef_4d
    return dropdims(sum(curve; dims = 4); dims = 4)
end

function curve2coef(
        b,
        x,
        y,
        grid,
        σ,
        scale;
        init = false,
        ε = 1.0f-4
    )
    B = b(x, grid, σ, scale; init = init)

    A, b_vec = regularize(
        PermutedDimsArray(B, (2, 3, 1)),
        PermutedDimsArray(y, (3, 2, 1)),
        b;
        ε = ε,
        init = init
    )

    A, b_vec, P = forward_elimination(A, b_vec, b; ε = ε)
    coef = dropdims(backward_substitution(A, b_vec, b, P); dims = 2)
    return PermutedDimsArray(coef, (3, 2, 1)) .* 1.0f0
end

## FFT basis functions ###
struct FFT_basis <: AbstractBasis
    I
    O
    G
    S
end

function (b::FFT_basis)(
        x,
        grid,
        σ
    )
    I, G, S = b.I, b.G, b.S

    x_3d = PermutedDimsArray(view(x, :, :, :), (1, 3, 2))
    freq = x_3d .* grid .* Float32(2π) .* σ
    return cos.(freq), sin.(freq)
end

function coef2curve_FFT(
        b,
        x_eval,
        grid,
        coef,
        σ,
    )
    I, O, G, S = b.I, b.O, b.G, b.S

    even, odd = b(x_eval, grid, σ)
    even = PermutedDimsArray(view(even, :, :, :, :), (1, 4, 3, 2))
    odd = PermutedDimsArray(view(odd, :, :, :, :), (1, 4, 3, 2))
    even_coef = coef[:, :, 1:1, :]
    odd_coef = coef[:, :, 2:2, :]
    return dropdims(
        sum(
            even .* even_coef .+ odd .* odd_coef; dims = 4
        ); dims = 4
    )
end

end
