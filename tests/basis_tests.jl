using Test, Random, LinearAlgebra

ENV["DEVICE"] = "cpu" # Don't change

include("../src/utils.jl")
using .Utils

include("../src/KAEM/kan/spline_bases.jl")
using .spline_functions

b, o, i, g, degree, σ, scale = 100, 8, 7, 2, 2, [one(Float32)], [one(Float32)]

function test_extend_grid()
    Random.seed!(42)
    grid = rand(Float32, i, g)
    extended_grid = extend_grid(grid; k_extend = degree)
    return @test size(extended_grid, 2) == size(grid, 2) + 2 * degree
end

function test_B_spline_basis()
    Random.seed!(42)
    x_eval = rand(Float32, i, b)
    Random.seed!(42)
    grid = rand(Float32, i, g)
    extended_grid = extend_grid(grid; k_extend = degree)
    coef = rand(Float32, i, o, g + degree - 1)

    basis_function = B_spline_basis(degree, i, o, size(extended_grid, 2), b)

    y = coef2curve_Spline(basis_function, x_eval, extended_grid, coef, σ, scale)
    @test size(y) == (i, o, b)
    @test !any(isnan.(y))

    recovered_coef = curve2coef(basis_function, x_eval, y, extended_grid, σ, scale; init = true, ε = 1.0f-4)
    @test size(recovered_coef) == size(coef)
    y_reconstructed =
        coef2curve_Spline(basis_function, x_eval, extended_grid, recovered_coef, σ, scale)
    return @test norm(y - y_reconstructed) < 1.0f-1
end

function test_RBF_basis()
    Random.seed!(42)
    x_eval = rand(Float32, i, b)
    Random.seed!(42)
    grid = rand(Float32, i, g)
    coef = rand(Float32, i, o, g)

    basis_function = RBF_basis(i, o, g, b)

    y = coef2curve_Spline(basis_function, x_eval, grid, coef, σ, scale)
    @test size(y) == (i, o, b)
    @test !any(isnan.(y))

    recovered_coef = curve2coef(basis_function, x_eval, y, grid, σ, scale; init = true, ε = 1.0f-4)
    @test size(recovered_coef) == size(coef)
    y_reconstructed = coef2curve_Spline(basis_function, x_eval, grid, recovered_coef, σ, scale)
    return @test norm(y - y_reconstructed) < 1.0f-1
end

function test_RSWAF_basis()
    Random.seed!(42)
    x_eval = rand(Float32, i, b)
    Random.seed!(42)
    grid = rand(Float32, i, g)
    coef = rand(Float32, i, o, g)

    basis_function = RSWAF_basis(i, o, g, b)

    y = coef2curve_Spline(basis_function, x_eval, grid, coef, σ, scale)
    @test size(y) == (i, o, b)
    @test !any(isnan.(y))

    recovered_coef = curve2coef(basis_function, x_eval, y, grid, σ, scale; init = true, ε = 1.0f-4)
    @test size(recovered_coef) == size(coef)
    y_reconstructed = coef2curve_Spline(basis_function, x_eval, grid, recovered_coef, σ, scale)
    return @test norm(y - y_reconstructed) < 1.0f-1
end

function test_FFT_basis()
    Random.seed!(42)
    x_eval = rand(Float32, i, b)
    Random.seed!(42)
    grid = rand(Float32, i, g)
    coef = rand(Float32, i, o, 2, g)

    basis_function = FFT_basis(i, o, g, b)

    y = coef2curve_FFT(basis_function, x_eval, grid, coef, σ)
    @test size(y) == (i, o, b)
    return @test !any(isnan.(y))
end

function test_Cheby_basis()
    Random.seed!(42)
    x_eval = rand(Float32, i, b)
    Random.seed!(42)
    grid = rand(Float32, i, g)
    coef = rand(Float32, i, o, degree + 1)

    basis_function = Cheby_basis(degree, i, o, b)

    y = coef2curve_Spline(basis_function, x_eval, grid, coef, σ, scale)
    @test size(y) == (i, o, b)
    @test !any(isnan.(y))

    recovered_coef = curve2coef(basis_function, x_eval, y, grid, σ, scale; init = true, ε = 1.0f-4)
    @test size(recovered_coef) == size(coef)
    y_reconstructed = coef2curve_Spline(basis_function, x_eval, grid, recovered_coef, σ, scale)
    return @test norm(y - y_reconstructed) < 1.0f-1
end

@testset "Spline Tests" begin
    test_extend_grid()
    # test_B_spline_basis()
    test_RBF_basis()
    test_RSWAF_basis()
    test_FFT_basis()
    test_Cheby_basis()
end
