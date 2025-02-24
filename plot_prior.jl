using JLD2, CUDA, Lux, LuxCUDA, CUDA, ComponentArrays, ConfParser, LaTeXStrings, Makie, GLMakie, Tullio, Random, Accessors

ENV["GPU"] = "true"

include("src/T-KAM/T-KAM.jl")
include("src/ML_pipeline/trainer.jl")
include("src/T-KAM/inversion_sampling.jl")
include("src/T-KAM/mixture_prior.jl")
include("src/utils.jl")
using .T_KAM_model
using .trainer
using .InverseSampling: prior_fwd
using .Utils: device, half_quant, hq
using .ebm_mix_prior: log_partition_function

file = "logs/gaussian_RBF/DARCY_FLOW_1/saved_model.jld2"
dataset_name = "DARCY_FLOW"
conf = ConfParse("config/darcy_flow_config.ini")
parse_conf!(conf)

# Components to plot (q, p)
plot_components = [(1,1), (6,2), (9,3), (3,4), (20,5)]
colours = [:red, :blue, :green, :purple, :orange]

saved_data = load(file)

ps = saved_data["params"] .|> half_quant |> device
st = saved_data["state"] |> hq |> device

rng = Random.seed!(1)
t = init_trainer(rng, conf, dataset_name; file_loc="garbage/")
prior = t.model.prior

ps = ps.ebm
st = st.ebm
t = nothing

a, b = minimum(st[Symbol("1")].grid), maximum(st[Symbol("1")].grid)
if b == prior.fcns_qp[Symbol("1")].grid_size
    a, b = prior.fcns_qp[Symbol("1")].grid_range
end
z = prior.quadrature_method == "trapezium" ? Float32.(range(a,b; length=1000)) |> device : (a + b) ./ 2 .+ (b - a) ./ 2 .* prior.nodes |> device
z =  prior.quadrature_method == "trapezium" ? repeat(z', prior.q_size, 1) : z
π_0 = prior.prior_type == "lognormal" ? prior.π_pdf(z, Float32(0.0001)) : prior.π_pdf(z)

f, st = prior_fwd(prior, ps, st, z)
alpha = softmax(ps[Symbol("α")]; dims=2) |> cpu_device()
f = exp.(f) .* permutedims(π_0[:,:,:], (1, 3, 2)) ./ exp.(first(log_partition_function(prior, ps, st)))
z, f = z |> cpu_device(), f |> cpu_device()

mkpath("figures/results/priors")

for (i, (q, p)) in enumerate(plot_components)
    fig = Makie.Figure(size=(1000, 1000),
                    ffont="Computer Modern", 
                    fontsize = 40,
                    backgroundcolor = :white,
                    show_axis = false,
                    show_grid = false,
                    show_axis_labels = false,
                    show_legend = false,
                    show_colorbar = false,
                )
    hue = round(alpha[q, p], digits=3)
    ax = Makie.Axis(fig[1, 1], title=L"Mixture component, ${\exp(f_{%$q,%$p}) \cdot \pi_0(z)} \; / \; {\textbf{Z}_{%$q,%$p}}$ \\ Mixture proportion $\alpha_{%$q,%$p} = %$hue$")

    band!(ax, z[q, :], 0 .* f[q, p, :], f[q, p, :], color=colours[i])
    lines!(ax, z[q, :], f[q, p, :], color=colours[i])
    # hidedecorations!(ax)
    # hidespines!(ax)
    save("figures/results/priors/$(dataset_name)_$(q)_$(p).png", fig)
end











