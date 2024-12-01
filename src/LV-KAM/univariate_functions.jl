module univariate_functions

using Lux, NNlib, LinearAlgebra, Random

include("spline_bases.jl")
using .spline_functions

SplineBasis_mapping = Dict(
    "B-spline" => B_spline_basis,
    "RBF" => RBF_basis,
    "RSWAF" => RSWAF_basis,
)

activation_mapping = Dict(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.tanh_fast,
    "sigmoid" => NNlib.sigmoid_fast,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu,
    "selu" => NNlib.selu,
    "tanh" => NNlib.tanh_fast,
    "silu" => x -> x .* NNlib.sigmoid_fast(x),
    "none" => x -> x .* 0f0
)

struct univariate_function <: Lux.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    spline_degree::Int
    base_activation::Function
    spline_function::Function
    grid::AbstractArray{Float32}
    grid_update_ratio::Float32
    grid_range::Tuple{Float32, Float32}
    ε_scale::Float32
    σ_base::Float32
    σ_spline::Float32
    init_α::Float32
    α_trainable::Bool
end

function init_function(
    in_dim::Int,
    out_dim::Int;
    spline_degree::Int=3,
    base_activation::String="silu",
    spline_function::String="B-spline",
    grid_size::Int=5,
    grid_update_ratio::Float32=2f-2,
    grid_range::Tuple{Float32, Float32}=(0f0, 1f0),
    ε_scale::Float32=1f-1,
    σ_base::Float32=nothing,
    σ_spline::Float32=1f0,
    init_α::Float32=1f0,
    α_trainable::Bool=true
)

    grid = Float32.(range(grid_range[1], grid_range[2], length=grid_size + 1)) |> collect |> x -> reshape(x, 1, length(x)) |> device
    grid = repeat(grid, latent_dim, 1) 
    grid = extend_grid(grid; k_extend=degree)  
    σ_base = isnothing(σ_base) ? ones(Float32, in_dim, out_dim) : σ_base
    base_activation = get(activation_mapping, base_activation, error("Invalid activation function, choices are: ", keys(activation_mapping)))
    spline_function = get(SplineBasis_mapping, spline_function, error("Invalid spline function, choices are: ", keys(SplineBasis_mapping)))
    
    return univariate_function(in_dim, out_dim, spline_degree, base_activation, spline_function, grid, grid_update_ratio, grid_range, ε_scale, σ_base, σ_spline, init_α, α_trainable)
end

function Lux.initialparameters(rng::AbstractRNG, l::univariate_function)
    ε = ((rand(rng, Float32, l.grid_size + 1, l.in_dim, l.out_dim) .- 0.5f0) .* l.ε_scale ./ l.grid_size)
    coef = curve2coef(l.grid[:, l.degree+1:end-l.degree] |> permutedims, ε, l.grid; k=l.degree, scale=m.init_α) 
    w_base = glorot_normal(rng, Float32, l.in_dim, l.out_dim) .* l.σ_base 
    w_sp = glorot_normal(rng, Float32, l.in_dim, l.out_dim) .* l.σ_sp 
    return m.a_trainable ? (w_base=w_base, w_sp=w_sp, coef=coef, basis_α=m.init_α) : (w_base=w_base, w_sp=w_sp, coef=coef)
end

function Lux.initialstates(rng::AbstractRNG, l::univariate_function)
    mask = ones(rng, Float32, l.in_dim, l.out_dim)
    return m.a_trainable ? (mask=mask) : (mask=mask, basis_α=m.init_α)
end

end