using Test, Random, LinearAlgebra, Lux

ENV["GPU"] = true

include("../LV-KAM/univariate_functions.jl")
include("../utils.jl")
using .univariate_functions
using .Utils

function test_fwd()
    Random.seed!(42)
    x = rand(Float32, 5, 3) |> device
    f = init_function(3, 2)

    Random.seed!(42)
    ps, st = Lux.setup(Random.GLOBAL_RNG, f)
    ps, st = ps |> device, st |> device

    y = fwd(f, ps, st, x)
    @test size(y) == (5, 3, 2)
end

function test_grid_update()
    Random.seed!(42)
    x = rand(Float32, 5, 3) |> device
    f = init_function(3, 2)
    ps, st = Lux.setup(Random.GLOBAL_RNG, f)
    ps, st = ps |> device, st |> device

    y = fwd(f, ps, st, x)
    grid, coef = update_lyr_grid(f, ps, st, x)
    @test size(grid) == (3, 12)
end

@testset "Univariate Funtion Tests" begin
    test_fwd()
    test_grid_update()
end