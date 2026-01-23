module UnivariateFunctions

export univariate_function, init_function, activation_mapping

using Accessors, ComponentArrays, NNlib
using Lux, NNlib, LinearAlgebra, Random

using ..Utils

include("spline_bases.jl")
using .spline_functions

include("wavelets.jl")
using .Wavelets

const SplineBasis_mapping = Dict(
    "B-spline" => (degree, in_dim, out_dim, grid_size, batch_size) -> B_spline_basis(degree, in_dim, out_dim, grid_size + 1, batch_size),
    "RBF" => (degree, in_dim, out_dim, grid_size, batch_size) -> RBF_basis(in_dim, out_dim, grid_size, batch_size),
    "RSWAF" => (degree, in_dim, out_dim, grid_size, batch_size) -> RSWAF_basis(in_dim, out_dim, grid_size, batch_size),
    "FFT" => (degree, in_dim, out_dim, grid_size, batch_size) -> FFT_basis(in_dim, out_dim, grid_size, batch_size),
    "Cheby" => (degree, in_dim, out_dim, grid_size, batch_size) -> Cheby_basis(degree, in_dim, out_dim, batch_size),
)

const Wavelet_mapping = Dict(
    "DoG" => DoGWavelet,
    "MH" => MHWavelet,
    "Morlet" => MorletWavelet,
    "Shannon" => ShannonWavelet
)

struct univariate_function{T <: Float32} <: Lux.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    base_activation::AbstractActivation
    basis_function::AbstractBasis
    spline_string::String
    spline_degree::Int
    init_grid::AbstractArray{T}
    grid_size::Int
    grid_update_ratio::T
    grid_range::Tuple{T, T}
    ε_scale::T
    σ_base::AbstractArray{T}
    σ_spline::T
    init_τ::AbstractArray{T}
    τ_trainable::Bool
    ε_ridge::T
end

function init_function(
        in_dim::Int,
        out_dim::Int;
        spline_degree::Int = 3,
        base_activation::AbstractString = "silu",
        spline_function::AbstractString = "RBF",
        grid_size::Int = 5,
        grid_update_ratio::T = 0.02f0,
        grid_range::Tuple{T, T} = (0.0f0, 1.0f0),
        ε_scale::T = 0.1f0,
        σ_base::AbstractArray{T} = [NaN32],
        σ_spline::T = 1.0f0,
        init_τ::T = 1.0f0,
        τ_trainable::Bool = true,
        ε_ridge::T = 1.0f-6,
        sample_size::Int = 1,
        wavelet::AbstractString = "Morlet"
    ) where {T <: Float32}
    spline_degree =
        (spline_function == "B-spline" || spline_function == "Cheby") ? spline_degree : 0
    grid_size = spline_function == "Cheby" || spline_function == "Wavelet" ? 1 : grid_size
    grid =
        spline_function == "FFT" ? collect(T, 0:grid_size) :
        range(grid_range[1], grid_range[2], length = grid_size + 1)
    grid = T.(grid) |> collect |> x -> reshape(x, 1, length(x))
    grid = repeat(grid, in_dim, 1)
    grid =
        !(spline_function == "Cheby" || spline_function == "FFT") ?
        extend_grid(grid; k_extend = spline_degree) : grid
    σ_base = any(isnan.(σ_base)) ? ones(T, in_dim, out_dim) : σ_base
    base_activation_obj =
        get(activation_mapping, base_activation, activation_mapping["silu"])

    # Extract concrete type for type parameter
    A = typeof(base_activation_obj)

    basis_function = nothing
    if spline_function == "Wavelet"
        basis_function = get(Wavelet_mapping, wavelet, MorletWavelet)()
    else
        initializer = get(
            SplineBasis_mapping,
            spline_function,
            (degree, I, O, G, S) -> RBF_basis(I, O, G, S)
        )
        basis_function = initializer(
            spline_degree,
            in_dim,
            out_dim,
            size(grid, 2),
            sample_size
        )
    end

    return univariate_function{T}(
        in_dim,
        out_dim,
        base_activation_obj,
        basis_function,
        spline_function,
        spline_degree,
        grid,
        size(grid, 2),
        grid_update_ratio,
        grid_range,
        ε_scale,
        σ_base,
        σ_spline,
        [init_τ],
        τ_trainable,
        ε_ridge,
    )
end

function Lux.initialparameters(
        rng::AbstractRNG,
        l::univariate_function{T},
    )::NamedTuple where {T <: Float32}

    if l.spline_string == "Wavelet"
        scale = glorot_uniform(rng, Float32, l.in_dim, l.out_dim)
        translation = glorot_normal(rng, Float32, l.in_dim, l.out_dim)
        weights = glorot_normal(rng, Float32, l.in_dim, l.out_dim)
        return (
            scale = scale,
            translation = translation,
            weights = weights,
            basis_τ = l.init_τ,
        )
    end

    w_base = glorot_normal(rng, Float32, l.in_dim, l.out_dim) .* l.σ_base
    w_sp = glorot_normal(rng, Float32, l.in_dim, l.out_dim) .* l.σ_spline

    coef = [0.0f0]
    if l.spline_string == "FFT"
        grid_norm_factor = collect(T, 1:(l.grid_size)) .^ 2
        coef =
            glorot_normal(rng, Float32, l.in_dim, l.out_dim, 2, l.grid_size) ./
            (sqrt(l.in_dim) .* permutedims(grid_norm_factor[:, :, :, :], [2, 3, 4, 1]))
    elseif !(l.spline_string == "Cheby")
        ε =
            (
            (rand(rng, Float32, l.in_dim, l.out_dim, l.grid_size) .- 0.5f0) .*
                l.ε_scale ./ l.grid_size
        )

        grid = l.init_grid
        scale = (maximum(grid) - minimum(grid)) / (size(grid, 2) - 1) |> Lux.f32
        coef = curve2coef(
            l.basis_function,
            l.init_grid[:, (l.spline_degree + 1):(end - l.spline_degree)],
            ε,
            l.init_grid,
            l.init_τ,
            scale,
            init = true,
            ε = l.ε_ridge
        )
    end

    if l.spline_string == "Cheby"
        return (
            coef = glorot_normal(rng, Float32, l.in_dim, l.out_dim, l.spline_degree + 1) .*
                (1 / (l.in_dim * (l.spline_degree + 1))),
            basis_τ = l.init_τ,
        )
    else
        return l.τ_trainable ?
            (w_base = w_base, w_sp = w_sp, coef = coef, basis_τ = l.init_τ) :
            (w_base = w_base, w_sp = w_sp, coef = coef)
    end
end

function Lux.initialstates(
        rng::AbstractRNG,
        l::univariate_function{T},
    )::NamedTuple where {T <: Float32}
    grid = l.init_grid
    scale = (maximum(grid) - minimum(grid)) / (size(grid, 2) - 1) |> Lux.f32

    # Domain
    min_z = repeat([first(l.grid_range)], l.in_dim)
    max_z = repeat([last(l.grid_range)], l.in_dim)

    return (
        grid = grid,
        basis_τ = l.init_τ,
        scale = [scale],
        min = min_z,
        max = max_z,
    )

end

function SplineMUL(
        l,
        ps,
        x,
        y,
    )
    x_act = l.base_activation(x)
    w_base, w_sp = ps.w_base, ps.w_sp
    I, S = l.basis_function.I, l.basis_function.S
    return w_base .* PermutedDimsArray(view(x_act, :, :, :), (1, 3, 2)) .+ w_sp .* y
end

function wavMUL(
        l,
        ps,
        x,
        basis_τ
    )
    x = PermutedDimsArray(view(x, :, :, :), (1, 3, 2))
    x = (x .- ps.translation) ./ ps.scale
    wavelet = l.basis_function(x, basis_τ)
    return wavelet .* ps.weights
end

function (l::univariate_function)(
        x,
        ps,
        st,
    )
    basis_τ = l.τ_trainable ? ps.basis_τ : st.basis_τ
    scale = st.scale
    l.spline_string == "Wavelet" && return wavMUL(l, ps, x, basis_τ)

    y =
        l.spline_string == "FFT" ?
        coef2curve_FFT(l.basis_function, x, st.grid, ps.coef, basis_τ) :
        coef2curve_Spline(l.basis_function, x, st.grid, ps.coef, basis_τ, scale)
    l.spline_string == "Cheby" && return y
    return SplineMUL(l, ps, x, y)
end

end
