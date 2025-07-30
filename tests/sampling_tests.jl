using Random, LinearAlgebra, Lux, ConfParser, ComponentArrays, CUDA, Test, ParallelStencil

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/utils.jl")
using .Utils

include("../src/T-KAM/T-KAM.jl")
using .T_KAM_model

include("../src/T-KAM/model_setup.jl")
using .ModelSetup

include("../src/T-KAM/ebm/inverse_transform.jl")
using .InverseTransformSampling: interp_kernel!

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

Random.seed!(42)

grid_size = 5
q_size, p_size, num_samples = 2, 2, 3

function test_interp_kernel()
    grid = pu(reshape(collect(full_quant, 0.0:0.25:1.0), 1, grid_size))
    grid = repeat(grid, p_size, 1)
    
    # Uniform CDF
    pdf_vals = @ones(q_size, p_size, grid_size) ./ grid_size
    cdf = cat(
        @zeros(q_size, p_size, 1),
        cumsum(pdf_vals, dims=3),
        dims=3
    )

    rand_vals = @rand(q_size, p_size, num_samples)
    
    z = @zeros(q_size, p_size, num_samples)
    
    @parallel (1:q_size, 1:p_size, 1:num_samples) interp_kernel!(
        z, cdf, grid, rand_vals, grid_size, half_quant(1e-8)
    )
    @test all((z .>= 0) .* (z .<= 1))
    
    # Kolmogorov-Smirnov test for uniformly distributed z at 5% significance
    z_flat = vec(Array(z))    
    sort!(z_flat)
    
    n = length(z_flat)
    theoretical_cdf = collect(1:n) ./ n
    D = maximum(abs.(theoretical_cdf .- (z_flat .- minimum(z_flat)) ./ (maximum(z_flat) - minimum(z_flat))))
    critical_value = 1.36 / sqrt(n)
    @test D < critical_value 
end

@testset "Inverse Transform Sampling" begin
    test_interp_kernel()
end
