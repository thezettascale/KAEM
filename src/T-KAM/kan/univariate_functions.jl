module UnivariateFunctions

export univariate_function, init_function, fwd, activation_mapping

using CUDA, KernelAbstractions, Accessors
using Lux, NNlib, LinearAlgebra, Random, LuxCUDA

include("spline_bases.jl")
include("../../utils.jl")
using .spline_functions
using .Utils: device, half_quant, full_quant

const SplineBasis_mapping = Dict(
    "B-spline" => B_spline_basis,
    "RBF" => RBF_basis,
    "RSWAF" => RSWAF_basis,
    "FFT" => FFT_basis,
    "Cheby" => Cheby_basis,
)

const activation_mapping = Dict(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.tanh_fast,
    "sigmoid" => NNlib.sigmoid_fast,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu,
    "selu" => NNlib.selu,
    "tanh" => NNlib.tanh_fast,
    "silu" => x -> x .* NNlib.sigmoid_fast(x),
    "elu" => NNlib.elu,
    "celu" => NNlib.celu,
    "none" => x -> x .* zero(half_quant),
)

struct univariate_function{T<:half_quant,U<:full_quant} <: Lux.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    spline_degree::Int
    base_activation::Function
    spline_string::String
    init_grid::AbstractArray{T}
    grid_size::Int
    grid_update_ratio::T
    grid_range::Tuple{T,T}
    ε_scale::T
    σ_base::AbstractArray{U}
    σ_spline::U
    init_τ::AbstractArray{U}
    τ_trainable::Bool
    basis_mul::Function
    coef2curve::Function
    curve2coef::Function
end

function ChebyMUL(l, ps, st, x::AbstractArray{T}) where {T<:half_quant}
    return y .* mask
end

function SplineMUL(l, ps, st, x::AbstractArray{T}) where {T<:half_quant}
    I, O, B = size(y)
    w_base, w_sp = ps.w_base, ps.w_sp
    base = l.base_activation(x)
    return w_base .* reshape(base, I, 1, B) .+ w_sp .* y .* mask
end

function init_function(
    in_dim::Int,
    out_dim::Int;
    spline_degree::Int = 3,
    base_activation::AbstractString = "silu",
    spline_function::AbstractString = "B-spline",
    grid_size::Int = 5,
    grid_update_ratio::T = half_quant(0.02),
    grid_range::Tuple{T,T} = (zero(half_quant), one(half_quant)),
    ε_scale::T = half_quant(0.1),
    σ_base::AbstractArray{U} = [full_quant(NaN)],
    σ_spline::U = one(full_quant),
    init_τ::U = one(full_quant),
    τ_trainable::Bool = true,
) where {T<:half_quant,U<:full_quant}
    spline_degree = spline_function == "B-spline" ? spline_degree : 0
    grid_size = spline_function == "Cheby" ? 1 : grid_size
    grid =
        spline_function == "FFT" ? collect(T, 0:grid_size) :
        range(grid_range[1], grid_range[2], length = grid_size + 1)
    grid = T.(grid) |> collect |> x -> reshape(x, 1, length(x)) |> device
    grid = repeat(grid, in_dim, 1)
    grid = extend_grid(grid; k_extend = spline_degree)
    σ_base = any(isnan.(σ_base)) ? ones(U, in_dim, out_dim) : σ_base
    base_activation =
        get(activation_mapping, base_activation, x -> x .* NNlib.sigmoid_fast(x))

    basis_fcn = get(SplineBasis_mapping, spline_function, B_spline_basis)
    basis_mul = spline_function == "Cheby" ? ChebyMUL : SplineMUL
    coef2curve =
        basis_fcn == FFT_basis ?
        (x, g, c, σ) -> coef2curve_FFT(x, g, c, σ; k = spline_degree) :
        (x, g, c, σ) ->
            coef2curve_Spline(x, g, c, σ; k = spline_degree, basis_function = basis_fcn)
    curve2coef =
        (x, g, y, σ) ->
            curve2coef(x, y, g, σ; k = spline_degree, basis_function = basis_fcn)

    return univariate_function(
        in_dim,
        out_dim,
        spline_degree,
        base_activation,
        spline_function,
        grid,
        grid_size,
        grid_update_ratio,
        grid_range,
        ε_scale,
        σ_base,
        σ_spline,
        [init_τ],
        τ_trainable,
        basis_mul,
        coef2curve,
        curve2coef,
    )
end

function Lux.initialparameters(rng::AbstractRNG, l::univariate_function)

    w_base = glorot_normal(rng, full_quant, l.in_dim, l.out_dim) .* l.σ_base
    w_sp = glorot_normal(rng, full_quant, l.in_dim, l.out_dim) .* l.σ_spline

    coef = nothing
    if l.spline_string == "FFT"
        grid_norm_factor = collect(1:(l.grid_size+1)) .^ 2
        coef =
            glorot_normal(rng, full_quant, 2, l.in_dim, l.out_dim, l.grid_size+1) ./
            (sqrt(l.in_dim) .* permutedims(grid_norm_factor[:, :, :, :], [2, 3, 4, 1]))
    elseif !(l.spline_string == "Cheby")
        ε =
            (
                (
                    rand(rng, half_quant, l.in_dim, l.out_dim, l.grid_size + 1) .-
                    half_quant(0.5)
                ) .* l.ε_scale ./ l.grid_size
            ) |> device
        coef = cpu_device()(
            l.curve2coef(
                l.init_grid[:, (l.spline_degree+1):(end-l.spline_degree)],
                ε,
                l.init_grid,
                device(half_quant.(l.init_τ)),
            ),
        )
    end

    if l.spline_string == "Cheby"
        return (
            coef = glorot_normal(rng, full_quant, l.in_dim, l.out_dim, l.spline_degree) .*
                   (1 / (l.in_dim * (l.spline_degree + 1))),
            basis_τ = l.init_τ,
        )
    else
        return l.τ_trainable ?
               (w_base = w_base, w_sp = w_sp, coef = coef, basis_τ = l.init_τ) :
               (w_base = w_base, w_sp = w_sp, coef = coef)
    end
end

function Lux.initialstates(rng::AbstractRNG, l::univariate_function)
    mask = ones(half_quant, l.in_dim, l.out_dim)
    return l.τ_trainable ? (mask = mask, grid = l.init_grid) :
           (mask = mask, grid = l.init_grid, basis_τ = half_quant.(l.init_τ))
end

function fwd(l, ps, st, x::AbstractArray{T}) where {T<:half_quant}
    """
    Forward pass for the univariate function.

    Args:
        l: The univariate function layer.
        ps: The parameters of the layer.
        st: The states of the layer.
        x_p: The input, (b, i).

    Returns:
        The output, (b, i, o), containing all fcn_{q,p}(x_p)
    """

    coef, mask = ps.coef, st.mask
    τ = l.τ_trainable ? ps.basis_τ : st.basis_τ

    y = l.coef2curve(x, st.grid, coef, τ)

    return l.basis_mul(l, ps, st, y)
end

end
