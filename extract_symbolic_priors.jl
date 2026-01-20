using JLD2, Lux, ConfParser, Random, ComponentArrays, Accessors
using MLDataDevices: cpu_device

ENV["DEVICE"] = "cpu"
ENV["GPU"] = "false"

include("src/utils.jl")
using .Utils

include("src/KAEM/KAEM.jl")
using .KAEM_model

include("src/pipeline/trainer.jl")
using .trainer

include("src/KAEM/symbolic/func_lib.jl")
using .SymbolicLibrary

include("src/KAEM/symbolic/symbolic_func.jl")
using .SymbolicFunctions

include("src/KAEM/symbolic/fit.jl")
using .FitSymbolic

include("src/KAEM/symbolic/plot.jl")
using .PlotKAN

include("src/KAEM/symbolic/transfer.jl")
using .Transfer

include("plotting/model_utils.jl")
using .ModelUtils

const MODE_TRAIN_TYPE = Dict(
    "thermo" => ("Thermodynamic", "ULA"),
    "vanilla" => ("Vanilla", "importance"), # Change for importance
    "variational" => ("Vanilla", "amortized"),
)

function parse_jobs_file(jobs_file::String)
    jobs = Tuple{String, String}[]

    for line in eachline(jobs_file)
        stripped = strip(line)
        if isempty(stripped) || startswith(stripped, "#")
            continue
        end

        parts = split(stripped)
        if length(parts) >= 2
            dataset, mode = parts[1], parts[2]
            if !startswith(mode, "baseline") && mode != "tune"
                push!(jobs, (String(dataset), String(mode)))
            end
        end
    end

    return jobs
end

function construct_model_path(dataset::String, mode::String, conf::ConfParse)
    folder_type, train_type = MODE_TRAIN_TYPE[mode]

    # Prior structure
    prior_type = retrieve(conf, "EbmModel", "π_0")
    spline_fcn = retrieve(conf, "EbmModel", "spline_function")
    prior_spline_fcn = "$(prior_type)_$(spline_fcn)"

    # Model structure
    layer_widths = parse.(Int, retrieve(conf, "EbmModel", "layer_widths"))
    use_mixture = parse(Bool, retrieve(conf, "MixtureModel", "use_mixture_prior"))

    structure = if length(layer_widths) > 2
        "deep"
    elseif use_mixture
        "mixture"
    else
        "univariate"
    end

    # Build path
    if train_type == "importance"
        path = "logs/$(folder_type)/$(dataset)/$(train_type)/$(prior_spline_fcn)/$(structure)"
    else
        path = "logs/$(folder_type)/$(dataset)/$(train_type)/$(structure)"
    end

    return path
end

function save_symbolic_prior(
        symbolic_params::Dict,
        prior,
        ps_ebm,
        st_kan_ebm,
        st_lux_ebm,
        st_quad,
        conf::ConfParse,
        dataset::String,
        mode::String,
        output_dir::String
    )
    prior_type = retrieve(conf, "EbmModel", "π_0")
    spline_fcn = retrieve(conf, "EbmModel", "spline_function")

    save_path = joinpath(output_dir, "symbolic_prior_$(dataset)_$(mode).jld2")
    mkpath(output_dir)

    cpu = cpu_device()
    jldsave(
        save_path;
        symbolic_params = symbolic_params,
        prior_config = Dict(
            "π_0" => prior_type,
            "spline_function" => spline_fcn,
            "q_size" => prior.q_size,
            "p_size" => prior.p_size,
            "depth" => prior.depth,
        ),
        ps_ebm = cpu(ps_ebm),
        st_kan_ebm = cpu(st_kan_ebm),
        st_lux_ebm = cpu(st_lux_ebm),
        st_quad = cpu(st_quad),
        dataset = dataset,
        mode = mode,
    )

    println("  Saved symbolic prior to: $save_path")
    return save_path
end

function plot_prior_components(
        prior,
        ps_ebm,
        st_kan_ebm,
        st_lux_ebm,
        st_quad,
        dataset::String,
        mode::String,
        output_dir::String;
        sym_fitter = nothing,
    )
    Q, P = prior.q_size, prior.p_size
    max_components = min(Q * P, 6)
    plot_components = [(q, p) for q in 1:min(Q, 3) for p in 1:min(P, 3)][1:max_components]
    colours = [:red, :blue, :green, :orange, :purple, :cyan][1:max_components]

    plot_dir = joinpath(output_dir, "plots", "$(dataset)_$(mode)")
    mkpath(plot_dir)

    println("  Plotting prior to: $plot_dir")

    return plot_ebm!(
        prior,
        ps_ebm,
        st_kan_ebm,
        st_lux_ebm,
        st_quad;
        plot_components = plot_components,
        component_colours = colours,
        file_loc = plot_dir,
        prior_type = dataset,
        fcn_type = mode,
        show_formula = sym_fitter !== nothing,
        sym_fitter = sym_fitter,
    )
end

function main(jobs_file::String = "jobs.txt")
    println("="^60)
    println("Symbolic Prior Extraction")
    println("="^60)

    # Parse jobs
    jobs = parse_jobs_file(jobs_file)
    println("\nFound $(length(jobs)) KAEM jobs to process:")
    for (dataset, mode) in jobs
        println("  - $dataset $mode")
    end

    output_dir = "figures/symbolic_priors"
    mkpath(output_dir)

    rng = Random.MersenneTwister(42)

    for (i, (dataset, mode)) in enumerate(jobs)
        println("\n" * "-"^60)
        println("[$i/$(length(jobs))] Processing: $dataset - $mode")
        println("-"^60)

        if !haskey(DATASET_CONFIGS, dataset)
            @warn "Unknown dataset: $dataset, skipping"
            continue
        end

        conf = ConfParse(DATASET_CONFIGS[dataset])
        parse_conf!(conf)

        setup_symbolic_config!(conf)

        model_path = construct_model_path(dataset, mode, conf)
        saved_file = joinpath(model_path, "saved_model.jld2")
        println("  Model path: $model_path")

        result = load_saved_model(saved_file, conf, dataset, init_trainer; rng = rng)
        if result === nothing
            @warn "Could not load model, skipping"
            continue
        end

        sym_fitter = SymFitter(conf)

        symbolic_params = fit_symbolic_prior(
            result.prior,
            result.ps_ebm,
            result.st_kan_ebm,
            sym_fitter,
            symbol_map;
            rng = rng
        )

        save_symbolic_prior(
            symbolic_params,
            result.prior,
            result.ps_ebm,
            result.st_kan_ebm,
            result.st_lux_ebm,
            result.st_quad,
            conf,
            dataset,
            mode,
            output_dir
        )

        plot_prior_components(
            result.prior,
            result.ps_ebm,
            result.st_kan_ebm,
            result.st_lux_ebm,
            result.st_quad,
            dataset,
            mode,
            output_dir;
            sym_fitter = sym_fitter
        )

        println("  Done!")
    end

    println("\n" * "="^60)
    println("Extraction complete!")
    println("Outputs saved to: $output_dir")
    return println("="^60)
end

# Run
jobs_file = length(ARGS) > 0 ? ARGS[1] : "jobs.txt"
main(jobs_file)
