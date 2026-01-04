module SymbolicFunctions

export symbolic_function, init_symbolic_function, get_formula, print_formulas, params_from_fit

using Lux, Random, LinearAlgebra, Accessors

using ..Utils

struct symbolic_function{T <: Float32} <: Lux.AbstractLuxLayer
    in_dim::Int
    out_dim::Int
    funcs::NamedTuple
    func_names::NamedTuple
    grid_range::Tuple{Vector{T}, Vector{T}}
    init_α::Matrix{T}
    init_β::Matrix{T}
    init_w::Matrix{T}
    init_b::Matrix{T}
end

function init_symbolic_function(
        in_dim::Int,
        out_dim::Int,
        min_grid::AbstractVector{T},
        max_grid::AbstractVector{T},
        fit_dict::Dict,
        α::AbstractMatrix{T},
        β::AbstractMatrix{T},
        w::AbstractMatrix{T},
        b::AbstractMatrix{T}
    ) where {T <: Float32}

    funcs = NamedTuple()
    func_names = NamedTuple()
    for i in 1:in_dim, o in 1:out_dim
        key = "i=$i,o=$o"
        if haskey(fit_dict, key)
            name, _, func = fit_dict[key]
            @reset funcs[Symbol("i=$i,o=$o")] = func
            @reset func_names[Symbol("i=$i,o=$o")] = name
        else
            @reset funcs[Symbol("i=$i,o=$o")] = x -> x
            @reset func_names[Symbol("i=$i,o=$o")] = "x"
        end
    end

    return symbolic_function{T}(
        in_dim,
        out_dim,
        funcs,
        func_names,
        (min_grid, max_grid),
        α,
        β,
        w,
        b,
    )
end


function Lux.initialparameters(
        rng::AbstractRNG,
        l::symbolic_function{T},
    )::NamedTuple where {T <: Float32}
    return (
        α = l.init_α,
        β = l.init_β,
        w = l.init_w,
        b = l.init_b,
    )
end

function Lux.initialstates(
        rng::AbstractRNG,
        l::symbolic_function{T},
    )::NamedTuple where {T <: Float32}
    return (min = first(l.grid_range), max = last(l.grid_range))
end

function (l::symbolic_function{T})(
        x::AbstractArray,
        ps,
        st,
    ) where {T <: Float32}

    α, β, w, b = ps.α, ps.β, ps.w, ps.b
    I, O = l.in_dim, l.out_dim

    z = α .* x .+ β
    for i in 1:I
        for o in 1:O
            func = l.funcs[Symbol("i=$i,o=$o")]
            z[i, o, :] = func(z[i, o, :])
        end
    end

    y = w .* z .+ b

    return y
end

function get_formula(
        l::symbolic_function,
        ps,
        i::Int,
        o::Int,
    )::String
    α, β, w, b = ps.α[i, o], ps.β[i, o], ps.w[i, o], ps.b[i, o]
    func_name = l.func_names[Symbol("i=$i,o=$o")]

    inner = abs(β) < 1.0e-6 ? "$(round(α, digits = 3))x" : "$(round(α, digits = 3))x + $(round(β, digits = 3))"
    outer = func_name == "x" ? inner : "$func_name($inner)"

    if abs(w - 1.0) < 1.0e-6 && abs(b) < 1.0e-6
        return outer
    elseif abs(b) < 1.0e-6
        return "$(round(w, digits = 3)) * $outer"
    else
        return "$(round(w, digits = 3)) * $outer + $(round(b, digits = 3))"
    end
end

function print_formulas(l::symbolic_function, ps)
    println("Symbolic Function Formulas:")
    println("-"^50)
    for i in 1:l.in_dim
        for o in 1:l.out_dim
            formula = get_formula(l, ps, i, o)
            println("  f[$i, $o](x) = $formula")
        end
    end
    return println("-"^50)
end

end
