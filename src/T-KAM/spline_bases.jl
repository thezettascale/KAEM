module spline_functions

export extend_grid, coef2curve, curve2coef, B_spline_basis, RBF_basis, RSWAF_basis, FFT_basis, Cheby_basis, Gottlieb_basis

using CUDA, KernelAbstractions
using LinearAlgebra, NNlib

include("../utils.jl")
using .Utils: removeNaN, device, half_quant, full_quant

method = get(ENV, "method", "B-spline") 

function extend_grid(grid::AbstractArray{T}; k_extend::Int=0) where {T<:half_quant}
    h = (grid[:, end] - grid[:, 1]) / (size(grid, 2) - 1)

    for i in 1:k_extend
        grid = hcat(grid[:, 1:1] .- h, grid)
        grid = hcat(grid, grid[:, end:end] .+ h)
    end
    
    return grid
end

function B_spline_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T};
    degree::Int=3, 
    σ::Union{T, AbstractArray{T}}=one(half_quant)
    ) where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    
    # 0-th degree
    if degree == 0
        grid_1 = grid[:, 1:end-1] 
        grid_2 = grid[:, 2:end] 
    
        # B0 is piecewise constant
        term1 = reshape(x, I, 1, S) .>= reshape(grid_1, I, G-1, 1)
        term2 = reshape(x, I, 1, S) .<  reshape(grid_2, I, G-1, 1)
        B = T.(term1 .* term2)

    else
        # k-th degree
        k = degree
        B = B_spline_basis(x, grid; degree=k-1)
        
        x = reshape(x, I, 1, S)

        numer1 = x .- grid[:, 1:(end - k - 1)]
        denom1 = grid[:, (k + 1):end-1] .- grid[:, 1:(end - k - 1)]
        numer2 = grid[:, (k + 2):end] .- x
        denom2 = grid[:, (k + 2):end] .- grid[:, 2:(end - k)]
        B_i1 = B[:, 1:end - 1, :]
        B_i2 = B[:, 2:end, :]

        B = (numer1 ./ denom1) .* B_i1 .+ (numer2 ./ denom2) .* B_i2
    end
    
    return B
end

function RBF_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T};
    degree::Int=3, 
    σ::Union{T, AbstractArray{T}}=one(half_quant)
    ) where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    σ = ((maximum(grid) - minimum(grid)) / (size(grid, 2) - 1)) * σ
    diff = reshape(x, I, 1, S) .- reshape(grid, I, G, 1)
    return exp.(-T(0.5) * (diff ./ σ).^2)  
end

function RSWAF_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T};
    degree::Int=3, 
    σ::Union{T, AbstractArray{T}}=one(half_quant)
    ) where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    diff = reshape(x, I, 1, S) .- reshape(grid, I, G, 1)
    diff = NNlib.tanh_fast(diff ./ σ)     
    return 1 .- diff.^2
end

function FFT_basis(
    x::AbstractArray{T},
    grid::AbstractArray{T};
    degree::Int=3, 
    σ::Union{T, AbstractArray{T}}=one(half_quant)
    ) where {T<:half_quant}
    I, S, G = size(x)..., size(grid, 2)
    x = reshape(x, I, 1, S) .* reshape(grid, I, G, 1)
    x = T(2π) .* x .* σ
    return cos.(x), sin.(x)
end

function Cheby_basis(
    x::AbstractArray{T},
    grid_::AbstractArray{T};
    degree::Int=3, 
    σ::Union{T, AbstractArray{T}}=half_quant(1.1) # IMPORTANT; to make sure acos is well-defined, set σ > 1, e.g. 1.1
    ) where {T<:half_quant}
    x = NNlib.tanh_fast(x) ./ σ 
    x = repeat(reshape(x, size(x)..., 1), 1, 1, degree+1)
    linspace = collect(0:degree) .|> T |> device
    return cos.(linspace' .* acos.(permutedims(x, [1, 3, 2])))
end

function Gottlieb_basis(
    x::AbstractArray{T},
    grid_::AbstractArray{T};
    degree::Int=3, 
    σ::Union{T, AbstractArray{T}}=one(half_quant)
    ) where {T<:half_quant}
    x = NNlib.sigmoid_fast(x)
    x = reshape(x, size(x)..., 1)
    B = ones(T, size(x, 1), size(x, 2), 1) |> device   
    B = cat(B, 2σ .* x, dims=3)
    for i in 3:degree+1
        y = 2(σ .+ (i-2)) .* x .* B[:, :, i-1] .- (σ .+ (2i-4)) .* B[:, :, i-2]
        B = cat(B, y, dims=3)
    end
    return permutedims(B, [1, 3, 2])
end

function coef2curve(
    x_eval::AbstractArray{T},
    grid::AbstractArray{T},
    coef::AbstractArray{T};
    k::Int=3, 
    scale::Union{T, AbstractArray{T}}=one(half_quant), 
    basis_function::Function=B_spline_basis
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
    spl = isnothing(basis_function) ? B_spline_basis(x_eval, grid; degree=k) : basis_function(x_eval, grid; degree=k, σ=scale)
    I, S, O, G = size(x_eval)..., size(coef)[2:3]...

    if !isa(spl, Tuple)
        spl = reshape(spl, I, G, 1, S) 
        coef = reshape(coef, I, G, O, 1)  
        return dropdims(sum(spl .* coef, dims=2), dims=2)
    else
        even, odd = spl
        even_coef, odd_coef = reshape(coef[1, :, :, :], I, G, O, 1), reshape(coef[2, :, :, :], I, G, O, 1)
        return dropdims(sum(even .* even_coef .+ odd .* odd_coef, dims=2), dims=2)
    end
end

function curve2coef(
    x::AbstractArray{T},
    y::AbstractArray{T},
    grid::AbstractArray{T};
    k::Int=3,
    scale::Union{T, AbstractArray{T}}=one(half_quant), 
    ε::U=full_quant(1f-4), 
    basis_function::Function=B_spline_basis
    ) where {T<:half_quant, U<:full_quant}
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

    B = basis_function(x, grid; degree=k, σ=scale) .|> full_quant
    G = size(B, 2)

    B = permutedims(B, [1, 3, 2]) # in_dim x b_size x n_grid

    coef = Array{U}(undef, J, O, G) |> device
    for i in 1:J
        for o in 1:O
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