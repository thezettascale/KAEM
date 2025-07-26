using Test, Random, LinearAlgebra
using NNlib: softmax

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/utils.jl")
using .Utils

include("../src/T-KAM/gen/resamplers.jl")
using .WeightResamplers

function test_systematic_resampler()
    Random.seed!(42)
    weights = rand(full_quant, 4, 4) |> pu
    ESS_bool = rand(Bool, 4) |> pu

    idxs = systematic_resampler(softmax(weights; dims = 2), ESS_bool, 4, 4)
    @test size(idxs) == (4, 4)
    @test !any(isnan, idxs)
end

function test_stratified_resampler()
    Random.seed!(42)
    weights = rand(full_quant, 4, 4) |> pu
    ESS_bool = rand(Bool, 4) |> pu

    idxs = stratified_resampler(softmax(weights; dims = 2), ESS_bool, 4, 4)
    @test size(idxs) == (4, 4)
    @test !any(isnan, idxs)
end

function test_residual_resampler()
    Random.seed!(42)
    weights = rand(full_quant, 4, 4) |> pu
    ESS_bool = rand(Bool, 4) |> pu

    idxs = residual_resampler(softmax(weights; dims = 2), ESS_bool, 4, 4)
    @test size(idxs) == (4, 4)
    @test !any(isnan, idxs)
end

@testset "Resampler Tests" begin
    test_systematic_resampler()
    test_stratified_resampler()
    test_residual_resampler()
end
