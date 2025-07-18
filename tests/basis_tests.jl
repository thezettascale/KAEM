using Test, Random, LinearAlgebra

ENV["GPU"] = false
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/T-KAM/kan/spline_bases.jl")
include("../src/utils.jl")
using .spline_functions
using .Utils

b, i, g, o, degree, σ = 5, 8, 7, 2, 2, device([one(half_quant)])

function test_extend_grid()
    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device
    extended_grid = extend_grid(grid; k_extend = degree)
    @test size(extended_grid, 2) == size(grid, 2) + 2 * degree
end

function test_B_spline_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device
    extended_grid = extend_grid(grid; k_extend = degree)
    coef = rand(half_quant, i, o, g + degree - 1) |> device

    y = coef2curve_Spline(
        x_eval,
        extended_grid,
        coef,
        σ;
        k = degree,
        basis_function = B_spline_basis,
    )
    @test size(y) == (i, o, b)
    @test !any(isnan.(y))

    recovered_coef =
        curve2coef(x_eval, y, extended_grid, σ; k = degree, basis_function = B_spline_basis)
    @test size(recovered_coef) == size(coef)
    y_reconstructed = coef2curve_Spline(
        x_eval,
        extended_grid,
        recovered_coef,
        σ;
        k = degree,
        basis_function = B_spline_basis,
    )
    @test norm(y - y_reconstructed) / norm(y) < half_quant(2)
end

function test_RBF_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device
    coef = rand(half_quant, i, o, g) |> device

    y = coef2curve_Spline(x_eval, grid, coef, σ; k = degree, basis_function = RBF_basis)
    @test size(y) == (i, o, b)
    @test !any(isnan.(y))

    recovered_coef = curve2coef(x_eval, y, grid, σ; k = degree, basis_function = RBF_basis)
    @test size(recovered_coef) == size(coef)
    y_reconstructed = coef2curve_Spline(
        x_eval,
        grid,
        recovered_coef,
        σ;
        k = degree,
        basis_function = RBF_basis,
    )
    @test norm(y - y_reconstructed) / norm(y) < half_quant(2)
end

function test_RSWAF_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device
    coef = rand(half_quant, i, o, g) |> device

    y = coef2curve_Spline(x_eval, grid, coef, σ; k = degree, basis_function = RSWAF_basis)
    @test size(y) == (i, o, b)
    @test !any(isnan.(y))

    recovered_coef =
        curve2coef(x_eval, y, grid, σ; k = degree, basis_function = RSWAF_basis)
    @test size(recovered_coef) == size(coef)
    y_reconstructed = coef2curve_Spline(
        x_eval,
        grid,
        recovered_coef,
        σ;
        k = degree,
        basis_function = RSWAF_basis,
    )
    @test norm(y - y_reconstructed) / norm(y) < half_quant(2)
end

function test_FFT_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device
    coef = rand(half_quant, 2, i, o, g) |> device

    y = coef2curve_Spline(x_eval, grid, coef, σ; k = degree, basis_function = FFT_basis)
    @test size(y) == (i, o, b)
    @test !any(isnan.(y))
end

function test_Cheby_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device
    coef = rand(half_quant, i, o, degree) |> device

    y = coef2curve_Spline(x_eval, grid, coef, σ; k = degree, basis_function = Cheby_basis)
    @test size(y) == (i, o, b)
    @test !any(isnan.(y))

    recovered_coef =
        curve2coef(x_eval, y, grid, σ; k = degree, basis_function = Cheby_basis)
    @test size(recovered_coef) == size(coef)
    y_reconstructed = coef2curve_Spline(
        x_eval,
        grid,
        recovered_coef,
        σ;
        k = degree,
        basis_function = Cheby_basis,
    )
    @test norm(y - y_reconstructed) / norm(y) < half_quant(2)
end

@testset "Spline Tests" begin
    test_extend_grid()
    test_B_spline_basis()
    test_RBF_basis()
    test_RSWAF_basis()
    test_FFT_basis()
    test_Cheby_basis()
end
