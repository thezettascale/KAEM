using Test, Random, LinearAlgebra

ENV["GPU"] = false

include("../LV-KAM/univariate_functions.jl")
include("../utils.jl")
using .spline_functions: extend_grid, B_spline_basis, RBF_basis, RSWAF_basis, coef2curve, curve2coef
using .Utils

Random.seed!(42)
b, i, g, o, degree, σ = 5, 3, 7, 2, 2, 1.0


function test_extend_grid()
    grid = rand(Float32, i, g) |> device
    extended_grid = extend_grid(grid; k_extend=degree)
    @test size(extended_grid, 2) == size(grid, 2) + 2 * degree
end

function test_B_spline_basis()
    x_eval = rand(Float32, b, i) |> device
    grid = rand(Float32, i, g) |> device
    extended_grid = extend_grid(grid; k_extend=degree)
    B = B_spline_basis(x_eval, extended_grid; degree=degree)
    @test size(B) == (b, i, g + degree - 1)
    @test !any(isnan.(B))
end

function test_RBF_basis()
    x_eval = rand(Float32, b, i) |> device
    grid = rand(Float32, i, g) |> device
    B_rbf = RBF_basis(x_eval, grid; σ=σ)
    @test size(B_rbf) == (b, i, g)
    @test !any(isnan.(B_rbf))
end

function test_RSWAF_basis()
    x_eval = rand(Float32, b, i) |> device
    grid = rand(Float32, i, g) |> device
    B_rswaf = RSWAF_basis(x_eval, grid; σ=σ)
    @test size(B_rswaf) == (b, i, g)
    @test !any(isnan.(B_rswaf))
end

function test_coef2curve()
    x_eval = rand(Float32, b, i) |> device
    grid = rand(Float32, i, g) |> device
    extended_grid = extend_grid(grid; k_extend=degree)
    coef = rand(Float32, i, o, g + degree - 1) |> device
    y_eval = coef2curve(x_eval, extended_grid, coef; k=degree, scale=σ)
    @test size(y_eval) == (b, i, o)
end

function test_curve2coef()
    x_eval = rand(Float32, b, i) |> device
    grid = rand(Float32, i, g) |> device
    extended_grid = extend_grid(grid; k_extend=degree)
    coef = rand(Float32, i, o, g + degree - 1) |> device
    y_eval = coef2curve(x_eval, extended_grid, coef; k=degree, scale=σ)
    recovered_coef = curve2coef(x_eval, y_eval, extended_grid; k=degree, scale=σ, ε=1e-4)
    @test size(recovered_coef) == size(coef)

    y_reconstructed = coef2curve(x_eval, extended_grid, recovered_coef; k=degree, scale=σ)
    @test norm(y_eval - y_reconstructed) / norm(y_eval) < 1e-3
end

@testset "Spline Functions Tests" begin
    test_extend_grid()
    test_B_spline_basis()
    test_RBF_basis()
    test_RSWAF_basis()
    test_coef2curve()
    test_curve2coef()
end
