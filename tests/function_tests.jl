using Test, Random, LinearAlgebra, Lux, ComponentArrays
using DifferentiationInterface, Enzyme

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"
ENV["AD_BACKEND"] = "ENZYME"

include("../src/T-KAM/kan/univariate_functions.jl")
include("../src/T-KAM/kan/grid_updating.jl")
include("../src/utils.jl")
using .UnivariateFunctions
using .GridUpdating: update_fcn_grid
using .Utils

test_backend = AutoEnzyme(;
    function_annotation = Enzyme.Duplicated,
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
)

function test_fwd()
    Random.seed!(42)
    x = rand(half_quant, 5, 3) |> device
    f = init_function(5, 2)

    Random.seed!(42)
    ps, st = Lux.setup(Random.GLOBAL_RNG, f)
    ps, st = ps |> device, st |> device

    y = fwd(f, ps, st, x)
    @test size(y) == (5, 2, 3)
end

function test_grid_update()
    Random.seed!(42)
    x = rand(half_quant, 5, 3) |> device
    f = init_function(5, 2)
    ps, st = Lux.setup(Random.GLOBAL_RNG, f)
    ps, st = ps |> device, st |> device

    y = fwd(f, ps, st, x)
    grid, coef = update_fcn_grid(f, ps, st, x)
    @test size(grid) == (5, 12)
end

function test_fwd_derivative()
    Random.seed!(42)
    x_eval = rand(half_quant, 5, 3) |> device
    f = init_function(5, 2)
    ps, st = Lux.setup(Random.GLOBAL_RNG, f)
    ps, st = ComponentArray(ps) |> device, st |> device
    g = p -> sum(fwd(f, p, st, x_eval))
    ∇ = gradient(g, test_backend, ps)
    @test size(∇) == size(ps)
    @test !any(isnan.(∇))
end

@testset "Univariate Funtion Tests" begin
    test_fwd()
    test_grid_update()
    test_fwd_derivative()
end
