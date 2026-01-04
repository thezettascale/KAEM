module KAN_Model

export KAN_Generator, init_KAN_Generator

using Lux, ComponentArrays, Accessors, Random, ConfParser

using ..Utils
using ..UnivariateFunctions
using ..SymbolicFunctions

struct BoolConfig <: AbstractBoolConfig
    layernorm::Bool
    batchnorm::Bool
end

struct KAN_Generator{T <: Float32} <: Lux.AbstractLuxLayer
    depth::Int
    Φ_fcns::Tuple{Vararg{Any}}
    layernorms::Tuple{Vararg{Lux.LayerNorm}}
    bool_config::BoolConfig
    x_shape::Tuple
    s_size::Int
end

# Outer constructor for Accessors.jl @reset
function KAN_Generator(
        depth,
        Φ_fcns,
        layernorms,
        bool_config,
        x_shape,
        s_size
    )
    return KAN_Generator{Float32}(
        depth,
        Φ_fcns,
        layernorms,
        bool_config,
        x_shape,
        s_size
    )
end

function init_KAN_Generator(
        conf::ConfParse,
        x_shape::Tuple,
        rng::AbstractRNG = Random.default_rng(),
    )

    prior_widths = (
        try
            parse.(Int, retrieve(conf, "EbmModel", "layer_widths"))
        catch
            parse.(Int, split(retrieve(conf, "EbmModel", "layer_widths"), ","))
        end
    )

    q_size = length(prior_widths) > 2 ? first(prior_widths) : last(prior_widths)

    widths = (
        try
            parse.(Int, retrieve(conf, "GeneratorModel", "widths"))
        catch
            parse.(Int, split(retrieve(conf, "GeneratorModel", "widths"), ","))
        end
    )

    widths = (widths..., prod(x_shape))
    first(widths) !== q_size && (
        error(
            "First expert Φ_hidden_widths must be equal to the hidden dimension of the prior.",
            widths,
            " != ",
            q_size,
        )
    )

    spline_degree = parse(Int, retrieve(conf, "GeneratorModel", "spline_degree"))
    layernorm_bool = parse(Bool, retrieve(conf, "GeneratorModel", "layernorm"))
    base_activation = retrieve(conf, "GeneratorModel", "base_activation")
    spline_function = retrieve(conf, "GeneratorModel", "spline_function")
    grid_size = parse(Int, retrieve(conf, "GeneratorModel", "grid_size"))
    grid_update_ratio =
        parse(Float32, retrieve(conf, "GeneratorModel", "grid_update_ratio"))
    grid_range = parse.(Float32, retrieve(conf, "GeneratorModel", "grid_range"))
    ε_scale = parse(Float32, retrieve(conf, "GeneratorModel", "ε_scale"))
    μ_scale = parse(Float32, retrieve(conf, "GeneratorModel", "μ_scale"))
    σ_base = parse(Float32, retrieve(conf, "GeneratorModel", "σ_base"))
    σ_spline = parse(Float32, retrieve(conf, "GeneratorModel", "σ_spline"))
    init_τ = parse(Float32, retrieve(conf, "GeneratorModel", "init_τ"))
    τ_trainable = parse(Bool, retrieve(conf, "GeneratorModel", "τ_trainable"))
    τ_trainable = spline_function == "B-spline" ? false : τ_trainable
    eps = parse(Float32, retrieve(conf, "TRAINING", "eps"))

    s_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    depth = length(widths) - 1

    initialize_function =
        (in_dim, out_dim, base_scale) -> init_function(
        in_dim,
        out_dim;
        spline_degree = spline_degree,
        base_activation = base_activation,
        spline_function = spline_function,
        grid_size = grid_size,
        grid_update_ratio = grid_update_ratio,
        grid_range = Tuple(grid_range),
        ε_scale = ε_scale,
        σ_base = base_scale,
        σ_spline = σ_spline,
        init_τ = init_τ,
        τ_trainable = τ_trainable,
        ε_ridge = eps,
        sample_size = s_size,
    )
    # Let Julia infer the concrete activation type from the elements we push
    Φ_functions = []
    layernorms = Vector{Lux.LayerNorm}(undef, 0)

    for i in eachindex(widths[1:(end - 1)])
        base_scale = (
            μ_scale * (1.0f0 / √(Float32(widths[i]))) .+
                σ_base .* (
                randn(rng, Float32, widths[i], widths[i + 1]) .* 2.0f0 .-
                    1.0f0
            ) .* (1.0f0 / √(Float32(widths[i])))
        )
        push!(Φ_functions, initialize_function(widths[i], widths[i + 1], base_scale))

        if layernorm_bool
            push!(layernorms, Lux.LayerNorm(widths[i]))
        end
    end

    A = length(Φ_functions) > 0 ? typeof(Φ_functions[1].base_activation) : AbstractActivation

    return KAN_Generator{Float32}(
        depth,
        Tuple(Φ_functions),
        Tuple(layernorms),
        BoolConfig(layernorm_bool, false),
        x_shape,
        s_size,
    )
end

function (gen::KAN_Generator)(
        ps,
        st_kan,
        st_lyrnorm,
        z,
    )
    """
    Generate data from the KAN likelihood model.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.

    Returns:
        The generated data.
    """
    z = dropdims(sum(z, dims = 2), dims = 2)

    # KAN functions
    state = (1, st_lyrnorm, z)
    while first(state) <= gen.depth
        i, st_lyrnorm_new, z_acc = state

        if gen.bool_config.layernorm
            z_acc, st_layer_new = Lux.apply(
                gen.layernorms[i],
                z_acc,
                ps.layernorm[symbol_map[i]],
                st_lyrnorm_new[symbol_map[i]]
            )
            @reset st_lyrnorm_new[symbol_map[i]] = st_layer_new
        end

        z_acc = Lux.apply(gen.Φ_fcns[i], z_acc, ps.fcn[symbol_map[i]], st_kan[symbol_map[i]])
        z_acc = dropdims(sum(z_acc, dims = 1); dims = 1)
        state = (i + 1, st_lyrnorm_new, z_acc)
    end

    _, st_lyrnorm_new, z = state
    return reshape(z, gen.x_shape..., gen.s_size), st_lyrnorm_new
end

end
