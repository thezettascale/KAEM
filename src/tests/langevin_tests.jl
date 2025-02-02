using CUDA, Lux, LuxCUDA
using Test, Random, LinearAlgebra, ConfParser, ComponentArrays, Plots, Distributions

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP16"

include("../T-KAM/T-KAM.jl")
include("../T-KAM/langevin_sampling.jl")
include("../utils.jl")
using .T_KAM_model
using .Utils: device, half_quant, full_quant
using .LangevinSampling

conf = ConfParse("src/tests/test_conf.ini")
parse_conf!(conf)
out_dim = parse(Int, retrieve(conf, "KAN_LIKELIHOOD", "output_dim"))

function plot_final_distribution()
    Random.seed!(42)
    dataset = randn(full_quant, 3, 50) 
    commit!(conf, "MALA", "use_langevin", "true")
    commit!(conf, "MALA", "iters", "150")
    commit!(conf, "TRAINING", "importance_sample_size", "100")
    model = init_T_KAM(dataset, conf)
    x_test = first(model.train_loader) |> device
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    ps, st = ComponentArray(ps) .|> half_quant |> device, st |> device

    output = first(model.posterior_sample(model, x_test, ps, st, 1)) |> cpu_device()
    
    final_samples = vec(output[2, :, :])
    
    # Plot the histogram of samples
    p = histogram(final_samples, bins=50, normalize=true, alpha=0.6, label="Samples")
    
    μ, σ = 0.0, 1.0  # Expected Gaussian parameters
    x = range(minimum(final_samples), maximum(final_samples), length=100)
    plot!(p, x, pdf.(Normal(μ, σ), x), lw=2, label="Expected Posterior")
    
    plot!(p, legend=:topleft, xlabel="z", ylabel="Density", title="Langevin Convergence")
    savefig(p, "figures/test/langevin_convergence.png")

    z_prior = first(model.prior.sample_z(model.prior, model.IS_samples, ps.ebm, st.ebm, 1)) |> cpu_device()

    @test size(z_prior) == size(output[1, :, :])
end

@testset "Langevin Tests" begin
    plot_final_distribution()
end