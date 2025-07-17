using Test, Random, LinearAlgebra

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/T-KAM/gen/resamplers.jl")
include("../src/utils.jl")
using .WeightResamplers
using .Utils

function test_systematic_resampler()
    Random.seed!(42)
    weights = rand(full_quant, 10, 10) |> device

    idxs, seed = importance_resampler(weights; resampler = systematic_resampler)
    @test size(idxs) == (10, 10)
end

function test_stratified_resampler()
    Random.seed!(42)
    weights = rand(full_quant, 10, 10) |> device

    idxs, seed = importance_resampler(weights; resampler = stratified_resampler)
    @test size(idxs) == (10, 10)
end

function test_residual_resampler()
    Random.seed!(42)
    weights = rand(full_quant, 10, 10) |> device

    idxs, seed = importance_resampler(weights; resampler = residual_resampler)
    @test size(idxs) == (10, 10)
end

@testset "Resampler Tests" begin
    test_systematic_resampler()
    test_stratified_resampler()
    test_residual_resampler()
end