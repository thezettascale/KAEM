using Test, Random, LinearAlgebra, Lux, ComponentArrays, Enzyme

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/T-KAM/kan/univariate_functions.jl")
include("../src/T-KAM/kan/grid_updating.jl")
include("../src/utils.jl")
using .UnivariateFunctions
using .GridUpdating: update_fcn_grid
using .Utils

function test_fwd()
    Random.seed!(42)
    x = rand(half_quant, 5, 3) |> device
    f = init_function(5, 2)

    Random.seed!(42)
    ps, st = Lux.setup(Random.GLOBAL_RNG, f)
    ps, st = ps |> ComponentArray |> device, st |> device

    y, st = Lux.apply(f, x, ps, st)
    @test size(y) == (5, 2, 3)
end

function test_grid_update()
    Random.seed!(42)
    x = rand(half_quant, 5, 3) |> device
    f = init_function(5, 2)
    ps, st = Lux.setup(Random.GLOBAL_RNG, f)
    ps, st = ps |> ComponentArray |> device, st |> device

    y, st = Lux.apply(f, x, ps, st)
    grid, coef = update_fcn_grid(f, ps, st, x)
    @test size(grid) == (5, 12)
end

function test_derivative()
    Random.seed!(42)
    x = rand(half_quant, 5, 3) |> device
    f = init_function(5, 2)
    ps, st = Lux.setup(Random.GLOBAL_RNG, f)
    ps, st = ps |> ComponentArray |> device, st |> device

    grads_ps = Enzyme.make_zero(ps)
    grads_x = Enzyme.make_zero(x)

    diff_fcn = (fcn, z, p, s) -> begin
        sum(first(Lux.apply(fcn, z, p, s)))
    end

    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.set_runtime_activity(Enzyme.Reverse)),
        diff_fcn,
        Enzyme.Active,
        Enzyme.Const(f),
        Enzyme.Duplicated(x, grads_x),
        Enzyme.Duplicated(ps, grads_ps),
        Enzyme.Const(st),
    )

    @test !all(iszero, grads_ps)
    @test !any(isnan, grads_ps)
    @test !all(iszero, grads_x)
    @test !any(isnan, grads_x)
end

@testset "Univariate Funtion Tests" begin
    # test_fwd()
    # test_grid_update()
    test_derivative()
end
