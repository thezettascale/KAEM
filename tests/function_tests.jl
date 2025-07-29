using Test, Random, LinearAlgebra, Lux, ComponentArrays, Enzyme

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/utils.jl")
using .Utils

include("../src/T-KAM/kan/univariate_functions.jl")
using .UnivariateFunctions

include("../src/T-KAM/kan/grid_updating.jl")
using .GridUpdating: update_fcn_grid

function test_fwd()
    Random.seed!(42)
    x = rand(half_quant, 5, 3) |> pu
    f = init_function(5, 2)

    Random.seed!(42)
    ps, st = Lux.setup(Random.GLOBAL_RNG, f)
    ps = ps |> ComponentArray |> pu
    st = st |> ComponentArray |> pu

    y = f(x, ps, st)
    @test size(y) == (5, 2, 3)
end

function test_grid_update()
    Random.seed!(42)
    x = rand(half_quant, 5, 3) |> pu
    f = init_function(5, 2)
    ps, st = Lux.setup(Random.GLOBAL_RNG, f)
    ps = ps |> ComponentArray |> pu
    st = st |> ComponentArray |> pu

    y = f(x, ps, st)
    grid, coef = update_fcn_grid(f, ps, st, x)
    @test size(grid) == (5, 12)
end

function test_derivative()
    Random.seed!(42)
    x = rand(half_quant, 5, 3) |> pu
    f = init_function(5, 2)
    ps, st = Lux.setup(Random.GLOBAL_RNG, f)
    ps = ps |> ComponentArray |> pu
    st = st |> ComponentArray |> pu
    ∇ = Enzyme.make_zero(ps)

    function closured_f(z, p, s)
        y = f(z, p, s)
        return sum(y)
    end

    Enzyme.autodiff_deferred(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(closured_f),
        Enzyme.Active,
        Enzyme.Const(x),
        Enzyme.Duplicated(ps, ∇),
        Enzyme.Const(st),
    )

    @test norm(∇) != 0
    @test !any(isnan, ∇)
end

@testset "Univariate Funtion Tests" begin
    test_fwd()
    test_grid_update()
    test_derivative()
end
