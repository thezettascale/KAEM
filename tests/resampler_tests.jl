using Test, Random, LinearAlgebra
using NNlib: softmax

ENV["DEVICE"] = "cpu"

include("../src/utils.jl")
using .Utils

include("../src/KAEM/gen/resamplers.jl")
using .WeightResamplers

st_rng = (resample_rv = rand(Float32, 4, 6, 1), extra = [0.0f0])

function test_systematic_resampler()
    Random.seed!(42)
    weights = randn(Float32, 4, 6) * 3
    ESS_bool = rand(Bool, 4)
    r = SystematicResampler(0.5, true)

    idxs = r(softmax(weights; dims = 2), st_rng)
    println(idxs)

    @test size(idxs) == (4, 6)
    @test !any(isnan, idxs)

    @test all(idxs .>= 1)
    return @test all(idxs .<= 6)
end

function test_residual_resampler()
    Random.seed!(42)
    weights = randn(Float32, 4, 6) * 3
    ESS_bool = rand(Bool, 4)
    r = ResidualResampler(0.5, true)

    idxs = r(softmax(weights; dims = 2), st_rng)
    println(idxs)
    @test size(idxs) == (4, 6)
    return @test !any(isnan, idxs)
end

@testset "Resampler Tests" begin
    test_systematic_resampler()
    test_residual_resampler()
end
