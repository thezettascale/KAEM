using Random, LinearAlgebra, Lux, ConfParser, ComponentArrays, CUDA, Test, ParallelStencil, Statistics

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
using .InverseTransformSampling: interp_kernel!, interp_kernel_mixture!

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, full_quant, 3)
else
    @init_parallel_stencil(Threads, full_quant, 3)
end

Random.seed!(42)

grid_size = 5
q_size, p_size, num_samples = 10, 20, 100

# For more comprehensive Gaussian testing
q_size_large, p_size_large, num_samples_large = 3, 4, 100

function test_interp_kernel_uniform()
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

function erf(x)
    return 2 / sqrt(π) * exp(-x^2) * (1 - exp(-x^2))
end

function test_interp_kernel_gaussian()
    μ, σ = 0.0, 1.0
    grid_range = μ .+ σ .* collect(-3.0:0.5:3.0)  # ±3σ range
    grid = pu(reshape(collect(full_quant, grid_range), 1, length(grid_range)))
    grid = repeat(grid, p_size, 1)
    
    pdf_vals = @zeros(q_size, p_size, length(grid_range))
    for i in 1:length(grid_range)
        pdf_vals[:, :, i] .+= exp(-0.5 * ((grid_range[i] - μ) / σ)^2) / (σ * sqrt(2π))
    end
    
    pdf_sum = sum(pdf_vals, dims=3)
    pdf_vals = pdf_vals ./ pdf_sum    
    cdf = cat(
        @zeros(q_size, p_size, 1),
        cumsum(pdf_vals, dims=3),
        dims=3
    )

    rand_vals = @rand(q_size, p_size, num_samples)
    
    z = @zeros(q_size, p_size, num_samples)
    
    @parallel (1:q_size, 1:p_size, 1:num_samples) interp_kernel!(
        z, cdf, grid, rand_vals, length(grid_range), half_quant(1e-8)
    )
    
    # 99.99% of the samples should be within 4σ of the mean
    @test all((z .>= μ - 4σ) .* (z .<= μ + 4σ))
    
    # Kolmogorov-Smirnov test for Gaussian distributed z at 5% significance
    z_flat = vec(Array(z))    
    z_standardized = (z_flat .- mean(z_flat)) ./ std(z_flat) # N(0, 1)    
    sort!(z_standardized)
    
    n = length(z_standardized)    
    theoretical_cdf = 0.5 * (1 .+ erf.(z_standardized ./ sqrt(2)))    
    empirical_cdf = collect(1:n) ./ n    
    D = maximum(abs.(empirical_cdf .- theoretical_cdf))
    critical_value = 1.36 / sqrt(n)
    
    @test D < critical_value    
    @test abs(mean(z_flat) - μ) < 0.5  # Mean should be close to μ
    @test abs(std(z_flat) - σ) < 0.5   # Std should be close to σ
end

@testset "Inverse Transform Sampling" begin
    test_interp_kernel_uniform()
    test_interp_kernel_gaussian()
end
