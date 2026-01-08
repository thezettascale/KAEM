module FitSymbolic

export SymFitter

using Statistics,
    LinearAlgebra,
    SymPyPythonCall,
    ComponentArrays,
    Optimization,
    OptimizationNLopt,
    NLopt,
    Random,
    ConfParser,
    Lux,
    Reactant

using ..Utils
using ..SymbolicLibrary

struct SymFitter
    lib
    num_points::Int
    max_iters::Int
    lb::Float32
    ub::Float32
end


function fit_affine(
        x::AbstractArray{T, 3},
        y::AbstractArray{T, 3},
        func::Function,
        I::Int,
        O::Int;
        rng::AbstractRNG = Random.MersenneTwister(1),
        param_lower_bound::T = -10.0f0,
        param_upper_bound::T = 10.0f0,
        max_iters::Int = 100
    )::Tuple{
        AbstractArray{T},
        AbstractArray{T},
        AbstractArray{T},
    } where {T <: Float32}
    α_init = glorot_normal(
        rng,
        Float32,
        I * O
    )

    β_init = glorot_normal(
        rng,
        Float32,
        I * O
    )

    params = vcat(α_init, β_init)

    function R2_cost(u, p; final = false)
        α, β = reshape(u[1:(I * O)], I, O), reshape(u[(I * O + 1):end], I, O)

        ŷ = func.(α .* x .+ β)
        μ_y = mean(y; dims = 3)

        RSS = sum((y - ŷ) .^ 2; dims = 3)
        TSS = sum((y .- μ_y) .^ 2; dims = 3)
        R2 = 1.0f0 .- (RSS ./ TSS)

        R2 = final ? R2 : -sum(R2)
        return R2
    end

    lb = fill(param_lower_bound, length(params))
    ub = fill(param_upper_bound, length(params))

    optf = Optimization.OptimizationFunction(R2_cost)
    prob = OptimizationProblem(optf, params; lb = lb, ub = ub)
    sol = solve(prob, NLopt.GN_ORIG_DIRECT_L(); maxiters = max_iters)
    u = sol.minimizer
    α, β = reshape(u[1:(I * O)], I, O), reshape(u[(I * O + 1):end], I, O)

    R2 = R2_cost(u, nothing; final = true)
    return dropdims(R2; dims = 3), α, β
end

function ols_wb(
        z::AbstractArray{T, 2},
        y::AbstractArray{T, 2}
    )::Tuple{
        AbstractVector{T},
        AbstractVector{T},
    } where {T <: Float32}
    z_mean = mean(z; dims = 1)
    y_mean = mean(y; dims = 1)
    z_centered = z .- z_mean
    y_centered = y .- y_mean

    num = sum(z_centered .* y_centered; dims = 1)
    den = sum(z_centered .^ 2; dims = 1)
    w = ifelse.(
        iszero.(den),
        zero(T),
        num ./ den
    )
    b = y_mean .- w .* z_mean
    return vec(w), vec(b)
end

function fit_symbolic(
        x::AbstractArray{T, 3},
        y::AbstractArray{T, 3},
        func::Function,
        I::Int,
        O::Int;
        rng::AbstractRNG = Random.MersenneTwister(1),
        param_lower_bound::T = -10.0f0,
        param_upper_bound::T = 10.0f0,
        max_iters::Int = 100
    )::Tuple{
        AbstractArray{T},
        AbstractArray{T},
        AbstractArray{T},
        AbstractArray{T},
        AbstractArray{T},
    } where {T <: Float32}
    R2, α, β = fit_affine(x, y, func, I, O; rng = rng, max_iters = max_iters)
    z = func.(α .* x .+ β)
    z, y = PermutedDimsArray(z, (3, 1, 2)), PermutedDimsArray(y, (3, 1, 2))
    w, b = ols_wb(reshape(z, :, I * O), reshape(y, :, I * O))
    w, b = reshape(w, I, O), reshape(b, I, O)
    return R2, α, β, w, b
end

function SymFitter(
        conf::ConfParse;
        symbolic_lib = SYMB_LIB
    )
    num_points = parse(Int, retrieve(conf, "SYMBOLIC_REG", "num_points_fitting"))
    max_iters = parse(Int, retrieve(conf, "SYMBOLIC_REG", "max_iters"))
    lb = parse(Float32, retrieve(conf, "SYMBOLIC_REG", "fit_lower_bound"))
    ub = parse(Float32, retrieve(conf, "SYMBOLIC_REG", "fit_upper_bound"))
    return SymFitter(
        symbolic_lib,
        num_points,
        max_iters,
        lb,
        ub
    )
end

function (sf::SymFitter)(
        ps,
        st_kan,
        kan_func;
        rng = Random.MersenneTwister(1),
    )
    """Finds best symbolic functions for each input and output dim"""
    in_min, in_max = st_kan.grid[:, 1], st_kan.grid[:, end]
    I = length(in_min)
    O = ndims(ps.coef) > 3 ? size(ps.coef, 3) : size(ps.coef, 2)

    z = zeros(Float32, I, 1, sf.num_points)
    for i in 1:I
        in_min, in_max = Float32(st_kan.grid[i, 1]), Float32(st_kan.grid[i, end])
        z[i, 1, :] = range(in_min, in_max, length = sf.num_points) |> collect
    end

    if ps.coef isa Reactant.ConcreteRArray
        z_input = z[:, 1, :] |> pu
        y = Reactant.@jit(kan_func(z_input, ps, st_kan)) |> Array
    else
        y = kan_func(z[:, 1, :], ps, st_kan)
    end

    i = 1
    z = repeat(z, 1, O, 1)
    R2_list = zeros(Float32, I, O, length(sf.lib))
    α_list = zeros(Float32, I, O, length(sf.lib))
    β_list = zeros(Float32, I, O, length(sf.lib))
    w_list = zeros(Float32, I, O, length(sf.lib))
    b_list = zeros(Float32, I, O, length(sf.lib))
    for (name, sym) in sf.lib
        R2, α, β, w, b = fit_symbolic(
            z,
            y,
            sym[1],
            I,
            O;
            rng = rng,
            param_lower_bound = sf.lb,
            param_upper_bound = sf.ub,
            max_iters = sf.max_iters
        )
        R2_list[:, :, i] .= R2
        α_list[:, :, i] .= α
        β_list[:, :, i] .= β
        w_list[:, :, i] .= w
        b_list[:, :, i] .= b
        i += 1
    end

    fit = Dict()
    α, β, w, b = (
        zero(α_list[:, :, 1]),
        zero(α_list[:, :, 1]),
        zero(α_list[:, :, 1]),
        zero(α_list[:, :, 1]),
    )
    for i in 1:I, o in 1:O
        sorted_R2s = sortperm(R2_list[i, o, :], rev = true)
        best = sorted_R2s[1]

        best_name, best_func_tuple = collect(sf.lib)[best]
        best_func = best_func_tuple[1]
        best_R2 = R2_list[i, o, best]
        α[i, o] = α_list[i, o, best]
        β[i, o] = β_list[i, o, best]
        w[i, o] = w_list[i, o, best]
        b[i, o] = b_list[i, o, best]

        fit = merge(fit, Dict("i=$i,o=$o" => (best_name, best_R2, best_func)))
    end

    return fit, α, β, w, b
end

end
