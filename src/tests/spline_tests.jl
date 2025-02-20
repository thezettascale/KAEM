using Test, Random, LinearAlgebra

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../T-KAM/spline_bases.jl")
include("../utils.jl")
using .spline_functions
using .Utils

b, i, g, o, degree, σ = 5, 3, 7, 2, 2, half_quant(1)

function test_extend_grid()
    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device

    extended_grid = extend_grid(grid; k_extend=degree)

    @test size(extended_grid, 2) == size(grid, 2) + 2 * degree
end

function test_B_spline_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device

    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device

    extended_grid = extend_grid(grid; k_extend=degree)
    B = B_spline_basis(x_eval, extended_grid; degree=degree)

    @test size(B) == (i, g + degree - 1, b)
    @test !any(isnan.(B))
end

function test_RBF_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device

    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device

    B_rbf = RBF_basis(x_eval, grid; σ=σ)

    @test size(B_rbf) == (i, g, b)
    @test !any(isnan.(B_rbf))
end

function test_RSWAF_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device

    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device

    B_rswaf = RSWAF_basis(x_eval, grid; σ=σ)

    @test size(B_rswaf) == (i, g, b)
    @test !any(isnan.(B_rswaf))
end

function test_FFT_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device

    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device

    B_fft, Bfft_2 = FFT_basis(x_eval, grid; σ=σ)

    @test size(B_fft) == (i, g, b)
    @test !any(isnan.(B_fft))
end

function test_Morlet_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device

    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device

    B_morlet = Morlet_basis(x_eval, grid; σ=σ)

    @test size(B_morlet) == (i, g, b)
    @test !any(isnan.(B_morlet))
end

function test_Shannon_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device

    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device

    B_shannon = Shannon_basis(x_eval, grid; σ=σ)

    @test size(B_shannon) == (i, g, b)
    @test !any(isnan.(B_shannon))
end

function test_coef2curve()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device

    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device

    Random.seed!(42)
    coef = rand(half_quant, i, o, g + degree - 1) |> device

    extended_grid = extend_grid(grid; k_extend=degree)

    y_eval = coef2curve(x_eval, extended_grid, coef; k=degree, scale=σ)
    @test size(y_eval) == (i, o, b)
end

function test_curve2coef()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device

    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device

    Random.seed!(42)
    coef = rand(half_quant, i, o, g + degree - 1) |> device

    extended_grid = extend_grid(grid; k_extend=degree) 
    
    y_eval = coef2curve(x_eval, extended_grid, coef; k=degree, scale=σ)
    recovered_coef = curve2coef(x_eval, y_eval, extended_grid; k=degree, scale=σ)
    @test size(recovered_coef) == size(coef)

    y_reconstructed = coef2curve(x_eval, extended_grid, recovered_coef; k=degree, scale=σ)
    @test norm(y_eval - y_reconstructed) / norm(y_eval) < half_quant(1e-2)
end

@testset "Spline Tests" begin
    test_extend_grid()
    test_B_spline_basis()
    test_RBF_basis()
    test_RSWAF_basis()
    test_FFT_basis()
    test_coef2curve()
    test_curve2coef()
end
