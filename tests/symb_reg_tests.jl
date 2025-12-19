using Test, Random, LinearAlgebra, Statistics, ComponentArrays

ENV["GPU"] = false # Don't change

include("../src/utils.jl")
using .Utils

include("../src/KAEM/symbolic/func_lib.jl")
using .SymbolicLibrary

include("../src/KAEM/symbolic/fit.jl")
using .FitSymbolic

include("../src/KAEM/KAEM.jl")
using .KAEM_model

include("../src/KAEM/symbolic/reg.jl")
using .Reg

Random.seed!(42)

function test_symbolic_functions()
    x = rand(Float32, 10)
    x_pos = abs.(x) .+ 1.0f-3

    # Test basic functions
    @test all(SYMB_LIB["x"][1](x) .== x)
    @test all(SYMB_LIB["x^2"][1](x) .== x .^ 2)
    @test all(SYMB_LIB["x^3"][1](x) .== x .^ 3)
    @test all(SYMB_LIB["abs"][1](x) .== abs.(x))
    @test all(SYMB_LIB["sin"][1](x) .== sin.(x))
    @test all(SYMB_LIB["cos"][1](x) .== cos.(x))

    y_th = 1.0f2
    result = SYMB_LIB["1/x"][3]((x_pos), (y_th))
    @test !any(isnan.(result[2]))
    @test !any(isinf.(result[2]))

    result = SYMB_LIB["sqrt"][3]((x_pos), (y_th))
    @test !any(isnan.(result[2]))
    @test !any(isinf.(result[2]))

    result = SYMB_LIB["log"][3]((x_pos), (y_th))
    @test !any(isnan.(result[2]))

    return @test length(SYMB_LIB) > 0
end

function test_ols_wb()
    Random.seed!(42)
    n = 100
    w_true = 2.0f0
    b_true = 1.0f0
    z = randn(Float32, n)
    y = w_true .* z .+ b_true .+ 0.1f0 .* randn(Float32, n)

    w, b = FitSymbolic.ols_wb(z, y)

    @test abs(w - w_true) < 0.5f0
    @test abs(b - b_true) < 0.5f0
    @test !isnan(w)
    @test !isnan(b)

    z_zero = zeros(Float32, n)
    w_zero, b_zero = FitSymbolic.ols_wb(z_zero, y)
    @test w_zero == 0.0f0
    @test !isnan(b_zero)
    return @test true
end

function test_fit_affine()
    Random.seed!(42)
    I, O, N = 3, 2, 50
    x = randn(Float32, I, O, N)

    func = x -> x
    α_true = ones(Float32, I, O)
    β_true = zeros(Float32, I, O)
    y = func.(α_true .* x .+ β_true)

    R2, α, β = FitSymbolic.fit_affine(x, y, func; max_iters = 50)

    @test size(R2) == (I, O)
    @test size(α) == (I, O)
    @test size(β) == (I, O)
    @test !any(isnan.(R2))
    @test !any(isnan.(α))
    @test !any(isnan.(β))
    @test all(R2 .> -1.0f0)

    func_sq = x -> x .^ 2
    y_sq = func_sq.(α_true .* x .+ β_true)
    R2_sq, α_sq, β_sq = FitSymbolic.fit_affine(x, y_sq, func_sq; max_iters = 50)

    @test size(R2_sq) == (I, O)
    @test !any(isnan.(R2_sq))

    return @test true
end

function test_fit_symbolic()
    Random.seed!(42)
    I, O, N = 2, 2, 30
    x = randn(Float32, I, O, N)

    func = x -> x
    R2, α, β, w, b = FitSymbolic.fit_symbolic(x, x, func, I, O; max_iters = 30)

    @test size(R2) == (I, O)
    @test size(α) == (I, O)
    @test size(β) == (I, O)
    @test size(w) == (I, O)
    @test size(b) == (I, O)
    @test !any(isnan.(R2))
    @test !any(isnan.(α))
    @test !any(isnan.(β))
    @test !any(isnan.(w))
    @test !any(isnan.(b))

    func_sq = x -> x .^ 2
    y_sq = func_sq.(x)
    R2_sq, α_sq, β_sq, w_sq, b_sq = FitSymbolic.fit_symbolic(x, y_sq, func_sq, I, O; max_iters = 30)

    @test size(R2_sq) == (I, O)
    @test !any(isnan.(R2_sq))

    return @test true
end

@testset "Symbolic Regression Tests" begin
    test_symbolic_functions()
    test_ols_wb()
    test_fit_affine()
    test_fit_symbolic()
end
