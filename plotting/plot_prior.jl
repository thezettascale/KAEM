using JLD2,
    Lux,
    ConfParser,
    Random,
    ComponentArrays

ENV["GPU"] = "false"

include("../src/utils.jl")
using .Utils

include("../src/KAEM/KAEM.jl")
using .KAEM_model

include("../src/pipeline/trainer.jl")
using .trainer

include("../src/KAEM/symbolic/plot.jl")
using .PlotKAN

for fcn_type in ["RBF"]
    for prior_type in ["gaussian"]
        for dataset_name in ["MNIST", "FMNIST"]
            file = "logs/Vanilla/$(dataset_name)/importance/$(prior_type)_$(fcn_type)/univariate/saved_model.jld2"

            conf_loc = Dict(
                # "DARCY_FLOW" => "config/darcy_flow_config.ini",
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

            ps_flat = saved_data["params"] .|> Float32
            st_kan = saved_data["kan_state"]
            st_lux = saved_data["lux_state"]

            rng = Random.MersenneTwister(1)
            t = init_trainer(rng, conf, dataset_name; file_loc = "garbage/")
            prior = t.model.prior

            # Reconstruct ComponentArray from flat array using model structure
            ps_template = Lux.initialparameters(rng, t.model)
            ps = ComponentArray(ps_flat, getaxes(ps_template))

            ps = ps.ebm
            st_quad = st_kan.quad
            st_kan = st_kan.ebm
            st_lux = st_lux.ebm
            t = nothing

            # Components to plot (q, p)
            plot_components = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
            colours = [:red, :blue, :green, :yellow, :orange, :pink]

            plot_ebm!(
                prior,
                ps,
                st_kan,
                st_lux,
                st_quad;
                plot_components = plot_components,
                component_colours = colours,
                file_loc = "figures/results/priors/$(dataset_name)",
                prior_type = prior_type,
                fcn_type = fcn_type,
            )

        end
    end
end
