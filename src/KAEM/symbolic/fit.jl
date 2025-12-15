module FitSymbolic

export symbolic_fit

using Statistics, LinearAlgebra, SymPyPythonCall, ComponentArrays

include("func_lib.jl")
using .SymbolicLibrary

struct SymFitter
    lib
    num_points::Int
end

function R2_cost(chain, x, y, func)
    α, β = u[:α], u[:β]

    ŷ = func(α .* x .+ β)
    μ_y = mean(y; dims = 3)

    RSS = sum((y - ŷ) .^ 2; dims = 3)
    TSS = sum((y - μ_ŷ) .^ 2; dims = 3)
    return 1.0f0 .- (RSS ./ TSS)
end

function SymFitter(
        conf::ConfParse;
        symbolic_lib = SYMB_LIB
    )

    num_points = parse(Int, retrieve(conf, "SYMBOLIC_REG", "num_points_fitting"))
    return SymFitter(
        symbolic_lib,
        num_points,
    )
end

function (sf::SymFitter)(
        ps,
        st,
        kan_func
    )
    """Finds best symbolic functions for each input and output dim"""
    in_min, in_max = st.grid[:, 1], st.grid[:, end]
    z = range(
        in_max,
        in_max,
        length = sf.num_points
    ) |> collect

    z = repeat(
        reshape(z, 1, 1, sf.num_points),
        kan_func.in_dim,
        kan_func.out_dim,
        1
    )
    y = kan_func(z[:, 1, :], ps, st)

    R2_list = []
    for (name, sym) in sf.lib
        R2, a, b, w, b = fit_affine(z, y, sym)
        push!(R2_list, R2)
    end

    # TODO: vectorize
    sorted_R2s = sortperm(R2s, rev = true)
    top_K = min(top_K, length(sorted_R2s))
    top_R2s = sorted_R2s[1:top_K]

    best_name, best_func = collect(symbolic_lib)[first(top_R2s)]
    best_R2 = R2s[first(top_R2s)]

    return best_name, best_func, best_R2
end

end
