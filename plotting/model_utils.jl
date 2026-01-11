module ModelUtils

export load_saved_model, setup_symbolic_config!, fit_symbolic_prior, DATASET_CONFIGS

using JLD2, Lux, ConfParser, Random, ComponentArrays
using MLDataDevices: cpu_device

const DATASET_CONFIGS = Dict(
    "MNIST" => "config/nist_config.ini",
    "FMNIST" => "config/nist_config.ini",
    "CIFAR10" => "config/cifar_config.ini",
    "SVHN" => "config/svhn_config.ini",
    "CELEBA" => "config/celeba_config.ini",
    "DARCY_FLOW" => "config/darcy_flow_config.ini",
    "PTB" => "config/text_config.ini",
    "SMS_SPAM" => "config/text_config.ini",
)

function setup_symbolic_config!(conf::ConfParse)
    commit!(conf, "SYMBOLIC_REG", "fit_symbolic_prior", "true")
    commit!(conf, "SYMBOLIC_REG", "fit_symbolic_llhood", "false")
    commit!(conf, "SYMBOLIC_REG", "num_points_fitting", "100")
    commit!(conf, "SYMBOLIC_REG", "max_iters", "50")
    commit!(conf, "SYMBOLIC_REG", "fit_lower_bound", "-10.0")
    commit!(conf, "SYMBOLIC_REG", "fit_upper_bound", "10.0")
    return conf
end

function load_saved_model(
        saved_file::String,
        conf::ConfParse,
        dataset::String,
        init_trainer_fn::Function;
        rng = Random.MersenneTwister(42)
    )
    if !isfile(saved_file)
        @warn "Model file not found: $saved_file"
        return nothing
    end

    println("  Loading: $saved_file")
    saved_data = load(saved_file)

    ps_flat = saved_data["params"] .|> Float32
    st_kan = saved_data["kan_state"]
    st_lux = saved_data["lux_state"]

    # Model structure
    t = init_trainer_fn(
        rng, conf, dataset;
        file_loc = "tmp/",
        save_model = false
    )
    model = t.model
    prior = model.prior

    # Reconstruct ComponentArray from flat array using model structure
    ps_template = Lux.initialparameters(rng, model)
    ps = ComponentArray(ps_flat, getaxes(ps_template))

    # Extract EBM-specific parts
    ps_ebm = ps.ebm
    st_quad = st_kan.quad
    st_kan_ebm = st_kan.ebm
    st_lux_ebm = st_lux.ebm

    return (
        model = model,
        ps = ps,
        st_kan = st_kan,
        st_lux = st_lux,
        prior = prior,
        ps_ebm = ps_ebm,
        st_kan_ebm = st_kan_ebm,
        st_lux_ebm = st_lux_ebm,
        st_quad = st_quad,
    )
end

function fit_symbolic_prior(
        prior,
        ps_ebm,
        st_kan_ebm,
        sym_fitter,
        symbol_map;
        rng = Random.MersenneTwister(42)
    )
    symbolic_params = Dict()

    println("  Fitting symbolic functions...")
    for i in 1:prior.depth
        layer = prior.fcns_qp[i]
        I, O = layer.in_dim, layer.out_dim

        ps_layer = cpu_device()(ps_ebm.fcn[symbol_map[i]])
        st_kan_layer = cpu_device()(st_kan_ebm[symbol_map[i]])

        fit_dict, α, β, w, b = sym_fitter(ps_layer, st_kan_layer, layer; rng = rng)

        # Store fit results (convert to regular arrays for saving)
        symbolic_params["layer_$(i)"] = Dict(
            "fit_dict" => fit_dict,
            "α" => Array(α),
            "β" => Array(β),
            "w" => Array(w),
            "b" => Array(b),
            "in_dim" => I,
            "out_dim" => O,
            "grid_min" => Array(st_kan_layer.grid[:, 1]),
            "grid_max" => Array(st_kan_layer.grid[:, end]),
        )

        # Print fit quality
        for (key, val) in fit_dict
            name, r2, _ = val
            println("    Layer $i, $key: $name (R² = $(round(r2, digits = 4)))")
        end
    end

    return symbolic_params
end

end
