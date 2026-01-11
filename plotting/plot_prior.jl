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

include("model_utils.jl")
using .ModelUtils

for fcn_type in ["RBF"]
    for prior_type in ["gaussian"]
        for dataset_name in ["MNIST", "FMNIST"]
            file = "logs/Vanilla/$(dataset_name)/importance/$(prior_type)_$(fcn_type)/univariate/saved_model.jld2"

            conf_loc = Dict(
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

            rng = Random.MersenneTwister(1)

            result = load_saved_model(file, conf, dataset_name, init_trainer; rng = rng)
            if result === nothing
                @warn "Could not load model from $file, skipping"
                continue
            end

            prior = result.prior
            ps = result.ps_ebm
            st_kan = result.st_kan_ebm
            st_lux = result.st_lux_ebm
            st_quad = result.st_quad

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
