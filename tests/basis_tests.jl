using Test, Random, LinearAlgebra

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/T-KAM/kan/spline_bases.jl")
include("../src/utils.jl")
using .spline_functions
using .Utils
using Enzyme

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
    B = B_spline_basis(x_eval, extended_grid, σ; degree = degree)
    @test size(B) == (i, g + degree - 1, b)
    @test !any(isnan.(B))

end

function test_B_spline_derivative()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    ∇ = Enzyme.make_zero(x_eval)
    grid = rand(half_quant, i, g) |> device
    extended_grid = extend_grid(grid; k_extend = degree)

    function fcn(z::AbstractArray{T}, g::AbstractArray{T}, sig::AbstractArray{T})::T
        sum(B_spline_basis(z, g, sig; degree = degree))
    end

    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        fcn,
        Enzyme.Active,
        Enzyme.Duplicated(x_eval, ∇),
        Enzyme.Const(extended_grid),
        Enzyme.Const(σ),
    )

    @test size(∇) == size(x_eval)
    @test !any(isnan.(∇))
end

function test_RBF_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device#
    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device
    B_rbf = RBF_basis(x_eval, grid, σ; degree = degree)
    @test size(B_rbf) == (i, g, b)
    @test !any(isnan.(B_rbf))
end

function test_RBF_derivative()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    ∇ = Enzyme.make_zero(x_eval)
    grid = rand(half_quant, i, g) |> device

    function fcn(z::AbstractArray{T}, g::AbstractArray{T}, sig::AbstractArray{T})::T
        sum(RBF_basis(z, g, sig; degree = degree))
    end

    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        fcn,
        Enzyme.Active,
        Enzyme.Duplicated(x_eval, ∇),
        Enzyme.Const(grid),
        Enzyme.Const(σ),
    )

    @test size(∇) == size(x_eval)
    @test !any(isnan.(∇))
end

function test_RSWAF_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device
    B_rswaf = RSWAF_basis(x_eval, grid, σ; degree = degree)
    @test size(B_rswaf) == (i, g, b)
    @test !any(isnan.(B_rswaf))
end

function test_RSWAF_derivative()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    ∇ = Enzyme.make_zero(x_eval)
    grid = rand(half_quant, i, g) |> device

    function fcn(z::AbstractArray{T}, g::AbstractArray{T}, sig::AbstractArray{T})::T
        sum(RSWAF_basis(z, g, sig; degree = degree))
    end

    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        fcn,
        Enzyme.Active,
        Enzyme.Duplicated(x_eval, ∇),
        Enzyme.Const(grid),
        Enzyme.Const(σ),
    )

    @test size(∇) == size(x_eval)
    @test !any(isnan.(∇))
end

function test_FFT_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device
    B_fft, Bfft_2 = FFT_basis(x_eval, grid, σ; degree = degree)
    @test size(B_fft) == (i, g, b)
    @test !any(isnan.(B_fft))
end

function test_FFT_derivative()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    ∇ = Enzyme.make_zero(x_eval)
    grid = rand(half_quant, i, g) |> device

    function fcn(z::AbstractArray{T}, g::AbstractArray{T}, sig::AbstractArray{T})::T
        sum(FFT_basis(z, g, sig; degree = degree))
    end

    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        fcn,
        Enzyme.Active,
        Enzyme.Duplicated(x_eval, ∇),
        Enzyme.Const(grid),
        Enzyme.Const(σ),
    )

    @test size(∇) == size(x_eval)
    @test !any(isnan.(∇))
end

function test_Cheby_basis()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device
    B_cheby = Cheby_basis(x_eval, grid, σ; degree = degree)
    @test size(B_cheby) == (i, degree + 1, b)
    @test !any(isnan.(B_cheby))
end

function test_Cheby_derivative()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    ∇ = Enzyme.make_zero(x_eval)
    grid = rand(half_quant, i, g) |> device

    function fcn(z::AbstractArray{T}, g::AbstractArray{T}, sig::AbstractArray{T})::T
        sum(Cheby_basis(z, g, sig; degree = degree))
    end

    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        fcn,
        Enzyme.Active,
        Enzyme.Duplicated(x_eval, ∇),
        Enzyme.Const(grid),
        Enzyme.Const(σ),
    )

    @test size(∇) == size(x_eval)
    @test !any(isnan.(∇))
end

function test_coef2curve()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device
    Random.seed!(42)
    coef = rand(half_quant, i, o, g + degree - 1) |> device
    extended_grid = extend_grid(grid; k_extend = degree)
    y_eval = coef2curve_Spline(
        x_eval,
        extended_grid,
        coef,
        σ;
        k = degree,
        basis_function = B_spline_basis,
    )
    @test size(y_eval) == (i, o, b)
end

function test_curve2coef()
    Random.seed!(42)
    x_eval = rand(half_quant, i, b) |> device
    Random.seed!(42)
    grid = rand(half_quant, i, g) |> device
    Random.seed!(42)
    coef = rand(half_quant, i, o, g + degree - 1) |> device
    extended_grid = extend_grid(grid; k_extend = degree)
    y_eval = coef2curve_Spline(
        x_eval,
        extended_grid,
        coef,
        σ;
        k = degree,
        basis_function = B_spline_basis,
    )
    recovered_coef = curve2coef(
        x_eval,
        y_eval,
        extended_grid,
        σ;
        k = degree,
        basis_function = B_spline_basis,
    )
    @test size(recovered_coef) == size(coef)
    y_reconstructed = coef2curve_Spline(
        x_eval,
        extended_grid,
        recovered_coef,
        σ;
        k = degree,
        basis_function = B_spline_basis,
    )
    @test norm(y_eval - y_reconstructed) / norm(y_eval) < half_quant(2)
end

@testset "Spline Tests" begin
    test_extend_grid()
    test_B_spline_basis()
    test_RBF_basis()
    test_RSWAF_basis()
    test_FFT_basis()
    test_Cheby_basis()
    test_coef2curve()
    test_curve2coef()
    # test_B_spline_derivative()
    test_RBF_derivative()
    test_RSWAF_derivative()
    test_FFT_derivative()
    test_Cheby_derivative()
end
