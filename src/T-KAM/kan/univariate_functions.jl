module UnivariateFunctions

export univariate_function, init_function, activation_mapping

using CUDA, Accessors, ComponentArrays
using Lux, NNlib, LinearAlgebra, Random, LuxCUDA, ParallelStencil

include("spline_bases.jl")
include("../../utils.jl")
using .spline_functions
using .Utils: device, half_quant, full_quant

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

const SplineBasis_mapping = Dict(
    "B-spline" => degree -> B_spline_basis(degree),
    "RBF" => scale -> RBF_basis(half_quant(scale)),
    "RSWAF" => degree -> RSWAF_basis(),
    "FFT" => degree -> FFT_basis(),
    "Cheby" => degree -> Cheby_basis(degree),
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
    base_activation::Any
    basis_function::Any
    spline_string::String
    spline_degree::Int
    init_grid::AbstractArray{T}
    grid_size::Int
    grid_update_ratio::T
    grid_range::Tuple{T,T}
    ε_scale::T
    σ_base::AbstractArray{U}
    σ_spline::U
    init_τ::AbstractArray{U}
    τ_trainable::Bool
end

## Stencils much faster than broadcast ##
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
    l::univariate_function{T,full_quant},
    ps::ComponentArray{T},
    x::AbstractArray{T},
    y::AbstractArray{T},
)::AbstractArray{T} where {T<:half_quant}
    I, O, B = size(y)
    base = l.base_activation(x)
    @parallel (1:I, 1:O, 1:B) spl_kernel!(y, base, ps.w_base, ps.w_sp)
    return y
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
    spline_degree =
        (spline_function == "B-spline" || spline_function == "Cheby") ? spline_degree : 0
    grid_size = spline_function == "Cheby" ? 1 : grid_size
    grid =
        spline_function == "FFT" ? collect(T, 0:grid_size) :
        range(grid_range[1], grid_range[2], length = grid_size + 1)
    grid = T.(grid) |> collect |> x -> reshape(x, 1, length(x)) |> device
    grid = repeat(grid, in_dim, 1)
    grid =
        !(spline_function == "Cheby" || spline_function == "FFT") ?
        extend_grid(grid; k_extend = spline_degree) : grid
    σ_base = any(isnan.(σ_base)) ? ones(U, in_dim, out_dim) : σ_base
    base_activation =
        get(activation_mapping, base_activation, x -> x .* NNlib.sigmoid_fast(x))

    initializer =
        get(SplineBasis_mapping, spline_function, degree -> B_spline_basis(degree))
    scale = (maximum(grid) - minimum(grid)) / (size(grid, 2) - 1)
    basis_function =
        spline_function == "RBF" ? initializer(scale) : initializer(spline_degree)

    return univariate_function(
        in_dim,
        out_dim,
        base_activation,
        basis_function,
        spline_function,
        spline_degree,
        grid,
        grid_size,
        grid_update_ratio,
        grid_range,
        ε_scale,
        σ_base,
        σ_spline,
        [init_τ],
        τ_trainable,
    )
end

function Lux.initialparameters(
    rng::AbstractRNG,
    l::univariate_function{T,U},
) where {T<:half_quant,U<:full_quant}

    w_base = glorot_normal(rng, U, l.in_dim, l.out_dim) .* l.σ_base
    w_sp = glorot_normal(rng, U, l.in_dim, l.out_dim) .* l.σ_spline

    coef = nothing
    if l.spline_string == "FFT"
        grid_norm_factor = collect(U, 1:(l.grid_size+1)) .^ 2
        coef =
            glorot_normal(rng, U, 2, l.in_dim, l.out_dim, l.grid_size+1) ./
            (sqrt(l.in_dim) .* permutedims(grid_norm_factor[:, :, :, :], [2, 3, 4, 1]))
    elseif !(l.spline_string == "Cheby")
        ε =
            (
                (rand(rng, T, l.in_dim, l.out_dim, l.grid_size + 1) .- T(0.5)) .*
                l.ε_scale ./ l.grid_size
            ) |> device
        coef = cpu_device()(
            curve2coef(
                l.basis_function,
                l.init_grid[:, (l.spline_degree+1):(end-l.spline_degree)],
                ε,
                l.init_grid,
                device(T.(l.init_τ)),
            ),
        )
    end

    if l.spline_string == "Cheby"
        return (
            coef = glorot_normal(rng, U, l.in_dim, l.out_dim, l.spline_degree + 1) .*
                   (1 / (l.in_dim * (l.spline_degree + 1))),
            basis_τ = l.init_τ,
        )
    else
        return l.τ_trainable ?
               (w_base = w_base, w_sp = w_sp, coef = coef, basis_τ = l.init_τ) :
               (w_base = w_base, w_sp = w_sp, coef = coef)
    end
end

# If GPU, return NamedTuple of CuPtrs, which will not work for Enzyme/Reactant GPU - must convert to ComponentArray
function Lux.initialstates(
    rng::AbstractRNG,
    l::univariate_function{T,U},
) where {T<:half_quant,U<:full_quant}
    return (grid = T.(cpu_device()(l.init_grid)), basis_τ = T.(l.init_τ))

end

function (l::univariate_function{T,U})(
    x::AbstractArray{T},
    ps::ComponentArray{T},
    st::ComponentArray{T}, # Unlike standard Lux, states are a ComponentArray
)::AbstractArray{T} where {T<:half_quant,U<:full_quant}
    basis_τ = l.τ_trainable ? ps.basis_τ : st.basis_τ
    y =
        l.spline_string == "FFT" ?
        coef2curve_FFT(l.basis_function, x, st.grid, ps.coef, basis_τ) :
        coef2curve_Spline(l.basis_function, x, st.grid, ps.coef, basis_τ)
    l.spline_string == "Cheby" && return y
    return SplineMUL(l, ps, x, y)
end

end
