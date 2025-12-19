module FitSymbolic

export symbolic_fit

using Statistics,
    LinearAlgebra,
    SymPyPythonCall,
    ComponentArrays,
    Optimization,
    Accessors,
    OptimizationNLopt,
    NLopt,
    Random,
    ConfParser,
    Lux

include("func_lib.jl")
using .SymbolicLibrary

struct SymFitter
    lib
    num_points::Int
    max_iters::Int
end


function fit_affine(
        x::AbstractArray{T, 3},
        y::AbstractArray{T, 3},
        func::Function;
        rng::AbstractRNG = Random.MersenneTwister(1),
        max_iters::Int = 100
    )::Tuple{
        AbstractArray{T},
        AbstractArray{T},
        AbstractArray{T},
    } where {T <: Float32}
    I, O = size(y, 1), size(y, 2)
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

    optf = Optimization.OptimizationFunction(R2_cost)
    prob = OptimizationProblem(optf, params)
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
        max_iters::Int = 100
    )::Tuple{
        AbstractArray{T},
        AbstractArray{T},
        AbstractArray{T},
        AbstractArray{T},
        AbstractArray{T},
    } where {T <: Float32}
    R2, α, β = fit_affine(x, y, func; rng = rng, max_iters = max_iters)
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
    return SymFitter(
        symbolic_lib,
        num_points,
        max_iters
    )
end

function (sf::SymFitter)(
        ps,
        st_kan,
        st_lux,
        kan_func
    )
    """Finds best symbolic functions for each input and output dim"""
    in_min, in_max = st.grid[:, 1], st.grid[:, end]
    I, O = kan_func.in_dim, kan_func.out_dim

    z = range(
        in_max,
        in_max,
        length = sf.num_points
    ) |> collect

    z = repeat(
        reshape(z, 1, 1, sf.num_points),
        I, O, 1
    )
    y = kan_func(z[:, 1, :], ps, st_kan)

    R2_list = zeros(Float32, I, O, length(sf.lib))
    i = 1
    for (name, sym) in sf.lib
        R2, a, b, w, b = fit_affine(z, y, sym)
        R2_list[:, :, i] .= R2
        i += 1
    end

    fit = Dict()
    for i in 1:I, o in 1:O
        sorted_R2s = sortperm(R2s[i, o, :], rev = true)
        best = sorted_R2s[1]

        best_name, best_func = collect(symbolic_lib)[best]
        best_R2 = R2s[best]

        @reset st_lux[Symbol("i=$i,o=$o")] = best_func
        fit = merge(fit, Dict("i=$i,o=$o" => (best_name, best_R2)))
    end

    return fit, st_lux
end

end
