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
    # Create grid from 0 to 1 with grid_size points
    grid = pu(reshape(collect(full_quant, 0.0:1.0/(grid_size-1):1.0), 1, grid_size))
    grid = repeat(grid, p_size, 1)
    
    # Uniform PDF values (equal probability for each grid interval)
    pdf_vals = @ones(q_size, p_size, grid_size-1) ./ (grid_size-1)
    
    # Create CDF by cumsum - this will have grid_size elements
    cdf = cat(
        @zeros(q_size, p_size, 1),
        cumsum(pdf_vals, dims=3),
        dims=3
    )

    rand_vals = @rand(q_size, p_size, num_samples)
    
    z = @zeros(q_size, p_size, num_samples)
    
    # Pass the actual grid size to the kernel
    @parallel (1:q_size, 1:p_size, 1:num_samples) interp_kernel!(
        z, cdf, grid, rand_vals, grid_size
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

# Abramowitz and Stegun approximation - (Maximum error: 1.5e-7)

function erf(x)
    
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    sign = x >= 0 ? 1 : -1
    x = abs(x)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x)
    
    return sign * y
end

function test_interp_kernel_gaussian()
    μ, σ = 0.0, 1.0
    grid_range = μ .+ σ .* collect(-3.0:0.5:3.0)  # ±3σ range
    grid = pu(reshape(collect(full_quant, grid_range), 1, length(grid_range)))
    grid = repeat(grid, p_size, 1)
    
    pdf_vals = @zeros(q_size, p_size, length(grid_range)-1)
    for i in 1:(length(grid_range)-1)
        mid_point = (grid_range[i] + grid_range[i+1]) / 2
        pdf_vals[:, :, i] .= exp(-0.5 * ((mid_point - μ) / σ)^2) / (σ * sqrt(2π))
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
        z, cdf, grid, rand_vals, length(grid_range)
    )
    
    # 99.99% of the samples should be within 4σ of the mean
    @test all((z .>= μ - 4σ) .* (z .<= μ + 4σ))
    
    z_flat = vec(Array(z))    
    sample_mean = mean(z_flat)
    sample_std = std(z_flat)
    
    @test abs(sample_mean - μ) < 0.5  # Mean should be close to μ
    @test abs(sample_std - σ) < 0.5   # Std should be close to σ
    
    if length(z_flat) > 50
        # Kolmogorov-Smirnov test for Gaussian distributed z at 5% significance
        z_standardized = (z_flat .- sample_mean) ./ sample_std
        sort!(z_standardized)
        
        n = length(z_standardized)
        theoretical_cdf = 0.5 * (1 .+ erf.(z_standardized ./ sqrt(2)))
        empirical_cdf = collect(1:n) ./ n
        
        D = maximum(abs.(empirical_cdf .- theoretical_cdf))
        critical_value = 1.36 / sqrt(n)
        
        @test D < critical_value
    end
end

@testset "Inverse Transform Sampling" begin
    test_interp_kernel_uniform()
    test_interp_kernel_gaussian()
end
