using JLD2,
    CUDA,
    Lux,
    LuxCUDA,
    CUDA,
    ComponentArrays,
    ConfParser,
    LaTeXStrings,
    Makie,
    GLMakie,
    Tullio,
    Random,
    Accessors

ENV["GPU"] = "true"

include("src/T-KAM/T-KAM.jl")
include("src/pipeline/trainer.jl")
include("src/T-KAM/ebm/ebm_model.jl")
include("src/utils.jl")
using .T_KAM_model
using .trainer
using .Utils: device, half_quant, hq

for fcn_type in ["RBF", "FFT"]
    for prior_type in ["gaussian", "lognormal", "uniform"]
        for dataset_name in ["DARCY_FLOW", "MNIST", "FMNIST"]
            file = "logs/$(prior_type)_$(fcn_type)/$(dataset_name)_1/saved_model.jld2"

            conf_loc = Dict(
                "DARCY_FLOW" => "config/darcy_flow_config.ini",
                "MNIST" => "config/nist_config.ini",
                "FMNIST" => "config/nist_config.ini",
            )[dataset_name]

            conf = ConfParse(conf_loc)
            parse_conf!(conf)
            commit!(conf, "EbmModel", "π_0", prior_type)

            if fcn_type == "RBF"
                commit!(conf, "EbmModel", "spline_function", "RBF")
                commit!(conf, "EbmModel", "base_activation", "silu")
            else
                commit!(conf, "EbmModel", "spline_function", "FFT")
                commit!(conf, "EbmModel", "base_activation", "none")
            end

            saved_data = load(file)

            ps = saved_data["params"] .|> half_quant |> device
            st = saved_data["state"] |> hq |> device

            rng = Random.MersenneTwister(1)
            t = init_trainer(rng, conf, dataset_name; file_loc = "garbage/", rng = rng)
            prior = t.model.prior

            ps = ps.ebm
            st = st.ebm
            t = nothing

            grid_range =
                Dict("uniform" => (0, 1), "lognormal" => (0, 3), "gaussian" => (-3, 3))[prior_type]

            a, b = minimum(st.fcn[1].grid; dims = 2), maximum(st.fcn[1].grid; dims = 2)
            if fcn_type == "FFT"
                a = fill(half_quant(first(grid_range)), size(a)) |> device
                b = fill(half_quant(last(grid_range)), size(b)) |> device
            end

            z = (a + b) ./ 2 .+ (b - a) ./ 2 .* device(prior.nodes)
            π_0 =
                prior.prior_type == "lognormal" ? prior.π_pdf(z, Float32(0.0001)) :
                prior.π_pdf(z)

            f, st = prior(ps, st, z)
            f = exp.(f) .* permutedims(π_0[:, :, :], (3, 1, 2))
            z, f, π_0 = z |> cpu_device(),
            softmax(f; dims = 3) |> cpu_device(),
            softmax(π_0; dims = 2) |> cpu_device()

            # Components to plot (q, p)
            plot_components = [(1, 1), (1, 2), (1, 3)]
            colours = [:red, :blue, :green]

            mkpath("figures/results/priors")

            for (i, (q, p)) in enumerate(plot_components)
                fig = Makie.Figure(
                    size = (1000, 1000),
                    ffont = "Computer Modern",
                    fontsize = 50,
                    backgroundcolor = :white,
                    show_axis = false,
                    show_grid = false,
                    show_axis_labels = false,
                    show_legend = false,
                    show_colorbar = false,
                )
                ax = Makie.Axis(
                    fig[1, 1],
                    title = L"Prior component, ${\exp(f_{%$q,%$p}(z)) \cdot \pi_0(z)} \; / \; {\textbf{Z}_{%$q,%$p}}$",
                )

                band!(
                    ax,
                    z[p, :],
                    0 .* f[q, p, :],
                    f[q, p, :],
                    color = (colours[i], 0.3),
                    label = L"{\exp(f_{%$q,%$p}(z)) \cdot \pi_0(z)}",
                )
                lines!(ax, z[p, :], f[q, p, :], color = colours[i])
                band!(
                    ax,
                    z[p, :],
                    0 .* f[q, p, :],
                    π_0[p, :],
                    color = (:gray, 0.2),
                    label = L"\pi_0(z)",
                )
                lines!(ax, z[p, :], π_0[p, :], color = (:gray, 0.8))
                y_min = minimum([minimum(f[q, p, :]), minimum(π_0[p, :])])
                ylims!(ax, y_min, nothing)
                axislegend(ax)
                hidedecorations!(ax)
                hidespines!(ax)
                save(
                    "figures/results/priors/$(dataset_name)_$(prior_type)_$(fcn_type)_$(q)_$(p).png",
                    fig,
                )
            end
        end
    end
end
