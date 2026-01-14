using Test, Random, LinearAlgebra, Lux, ComponentArrays, Reactant

ENV["DEVICE"] = "gpu"

include("../src/utils.jl")
using .Utils

include("../src/KAEM/kan/univariate_functions.jl")
using .UnivariateFunctions

include("../src/KAEM/kan/grid_updating.jl")
using .GridUpdating: update_fcn_grid

function test_fwd()
    Random.seed!(42)
    x = rand(Float32, 5, 3)
    f = init_function(5, 2; sample_size = 3)

    Random.seed!(42)
    ps, st = Lux.setup(Random.GLOBAL_RNG, f)

    compiled_f = Reactant.@compile f(x, ps, st)
    y = compiled_f(x, ps, st)
    return @test size(y) == (5, 2, 3)
end

function test_grid_update()
    Random.seed!(42)
    x = rand(Float32, 5, 3)
    f = init_function(5, 2; sample_size = 3)
    ps, st = Lux.setup(Random.GLOBAL_RNG, f)

    compiled_f = Reactant.@compile f(x, ps, st)
    y = compiled_f(x, ps, st)
    compiled_update = Reactant.@compile update_fcn_grid(f, ps, st, x)
    grid, coef = compiled_update(f, ps, st, x)
    return @test !all(st.grid .== Array(grid))
end

@testset "Univariate Funtion Tests" begin
    test_fwd()
    test_grid_update()
end
