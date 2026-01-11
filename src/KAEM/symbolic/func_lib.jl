module SymbolicLibrary

export SYMB_LIB

using LinearAlgebra

# Conditionally load SymPyPythonCall (not needed for core functionality)
const SYMPY_AVAILABLE = try
    @eval using SymPyPythonCall
    true
catch
    false
end

nan_to_num = function (x)
    x = ifelse.(isnan.(x), zero(x), x)
    x = ifelse.(isinf.(x), zero(x), x)
    return x
end

sign(x) = x < 0 ? -1.0f0 : (x > 0 ? 1.0f0 : 0.0f0)
safe_log(x) = x > 0 ? log(x) : (x < 0 ? complex(log(-x), Float32(π)) : -Inf32)

# Singularity protection functions
function f_inv((x), (y_th))
    x_th = (1 ./ y_th)
    return (x_th, (y_th ./ x_th) .* x .* (abs.(x) .< x_th) .+ nan_to_num(1 ./ x) .* (abs.(x) .>= x_th))
end

function f_inv2((x), (y_th))
    x_th = sqrt.(1 ./ y_th)
    return (x_th, y_th .* (abs.(x) .< x_th) .+ nan_to_num(1 ./ x .^ 2) .* (abs.(x) .>= x_th))
end

function f_inv3((x), (y_th))
    x_th = 1 ./ y_th .^ (1 / 3)
    return (x_th, y_th ./ x_th .* x .* (abs.(x) .< x_th) .+ nan_to_num(1 ./ x .^ 3) .* (abs.(x) .>= x_th))
end

function f_inv4((x), (y_th))
    x_th = 1 ./ y_th .^ (1 / 4)
    return (x_th, y_th .* (abs.(x) .< x_th) .+ nan_to_num(1 ./ x .^ 4) .* (abs.(x) .>= x_th))
end

function f_inv5((x), (y_th))
    x_th = 1 ./ y_th .^ (1 / 5)
    return (x_th, y_th ./ x_th .* x .* (abs.(x) .< x_th) .+ nan_to_num(1 ./ x .^ 5) .* (abs.(x) .>= x_th))
end

function f_sqrt((x), (y_th))
    x_th = 1 ./ y_th .^ 2
    return (x_th, x_th ./ y_th .* x .* (abs.(x) .< x_th) .+ nan_to_num(sqrt.(abs.(x)) .* sign.(x)) .* (abs.(x) .>= x_th))
end

f_power1d5((x), (y_th)) = abs.(x) .^ 1.5

function f_invsqrt((x), (y_th))
    x_th = 1 ./ y_th .^ 2
    return (x_th, y_th .* (abs.(x) .< x_th) .+ nan_to_num(1 ./ sqrt.(abs.(x))) .* (abs.(x) .>= x_th))
end

function f_log((x), (y_th))
    x_th = exp.(-y_th)
    return (x_th, -y_th .* (abs.(x) .< x_th) .+ nan_to_num(log.(abs.(x))) .* (abs.(x) .>= x_th))
end

function f_tan((x), (y_th))
    clip = x .% π
    delta = Float32(π / 2) .- atan.(y_th)
    return (delta, -y_th ./ delta .* (clip .- Float32(π / 2)) .* (abs.(clip .- Float32(π / 2)) .< delta) .+ nan_to_num(tan.(clip)) .* (abs.(clip .- Float32(π / 2)) .>= delta))
end

function f_arctanh((x), (y_th))
    delta = 1 .- tanh.(y_th) .+ 1.0f-4
    return (delta, y_th .* sign.(x) .* (abs.(x) .> 1 .- delta) .+ nan_to_num(atanh.(x)) .* (abs.(x) .<= 1 .- delta))
end

function f_arcsin((x), (y_th))
    return (nothing, Float32(π / 2) .* sign.(x) .* (abs.(x) .> 1) .+ nan_to_num(asin.(x)) .* (abs.(x) .<= 1))
end

function f_arccos((x), (y_th))
    return (nothing, Float32(π / 2) .* (1 .- sign.(x)) .* (abs.(x) .> 1) .+ nan_to_num(acos.(x)) .* (abs.(x) .<= 1))
end

function f_exp((x), (y_th))
    x_th = log.(y_th)
    return ((x_th, y_th .* (x .> x_th) .+ exp.(x) .* (x .<= x_th)))
end

const SYMB_LIB = Dict(
    "x" => (x -> x, x -> (x, 1), ((x), (y_th)) -> ((), x)),
    "x^2" => (x -> x .^ 2, x -> (x .^ 2, 2), ((x), (y_th)) -> ((), x .^ 2)),
    "x^3" => (x -> x .^ 3, x -> (x .^ 3, 3), ((x), (y_th)) -> ((), x .^ 3)),
    "x^4" => (x -> x .^ 4, x -> (x .^ 4, 3), ((x), (y_th)) -> ((), x .^ 4)),
    "x^5" => (x -> x .^ 5, x -> (x .^ 5, 3), ((x), (y_th)) -> ((), x .^ 5)),
    "1/x" => (x -> 1 ./ x, x -> (1 ./ x, 2), f_inv),
    "1/x^2" => (x -> 1 ./ x .^ 2, x -> (1 ./ x .^ 2, 2), f_inv2),
    "1/x^3" => (x -> 1 ./ x .^ 3, x -> (1 ./ x .^ 3, 3), f_inv3),
    "1/x^4" => (x -> 1 ./ x .^ 4, x -> (1 ./ x .^ 4, 4), f_inv4),
    "1/x^5" => (x -> 1 ./ x .^ 5, x -> (1 ./ x .^ 5, 5), f_inv5),
    "sqrt" => (x -> sqrt.(abs.(x)), x -> (sympy.sqrt.(x), 2), f_sqrt),
    "x^0.5" => (x -> sqrt.(abs.(x)), x -> (sympy.sqrt.(x), 2), f_sqrt),
    "x^1.5" => (x -> sqrt.(abs.(x)) .^ 3, x -> (sympy.sqrt.(x) .^ 3, 4), f_power1d5),
    "1/sqrt(x)" => (x -> 1 ./ sqrt.(abs.(x)), x -> (1 ./ sympy.sqrt.(x), 2), f_invsqrt),
    "1/x^0.5" => (x -> 1 ./ sqrt.(abs.(x)), x -> (1 ./ sympy.sqrt.(x), 2), f_invsqrt),
    "exp" => (x -> exp.(x), x -> (sympy.exp.(x), 2), f_exp),
    "log" => (x -> log.(abs.(x)), x -> (sympy.log.(abs.(x)), 2), f_log),
    "abs" => (x -> abs.(x), x -> (sympy.Abs.(x), 3), ((x), (y_th)) -> ((), abs.(x))),
    "sin" => (x -> sin.(x), x -> (sympy.sin.(x), 2), ((x), (y_th)) -> ((), sin.(x))),
    "cos" => (x -> cos.(x), x -> (sympy.cos.(x), 2), ((x), (y_th)) -> ((), cos.(x))),
    "tan" => (x -> tan.(x), x -> (sympy.tan.(x), 3), f_tan),
    "tanh" => (x -> tanh.(x), x -> (sympy.tanh.(x), 3), ((x), (y_th)) -> ((), tanh.(x))),
    "sgn" => (x -> sign.(x), x -> (sympy.sign.(x), 3), ((x), (y_th)) -> ((), sign.(x))),
    # "arcsin" => (x -> asin.(x), x -> (sympy.asin.(x), 4), f_arcsin),
    # "arccos" => (x -> acos.(x), x -> (sympy.acos.(x), 4), f_arccos),
    # "arctan" => (x -> atan.(x), x -> (sympy.atan.(x), 4), ((x), (y_th)) -> ((), atan.(x))),
    # "arctanh" => (x -> atanh.(x), x -> (sympy.atanh.(x), 4), f_arctanh),
    "0" => (x -> x * 0, x -> (x * 0, 0), ((x), (y_th)) -> ((), x * 0)),
    "gaussian" => (x -> exp.(-x .^ 2), x -> (sympy.exp.(-x .^ 2), 3), ((x), (y_th)) -> ((), exp.(-x .^ 2)))
)

end
