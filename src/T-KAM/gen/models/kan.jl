
module KAN_Model

export KAN_Generator, init_KAN_Generator

using CUDA, Lux, LuxCUDA, ComponentArrays, Accessors, Random, ConfParser

include("../../kan/univariate_functions.jl")
include("../../../utils.jl")
using .Utils: half_quant, full_quant, symbol_map
using .UnivariateFunctions

struct KAN_Generator{T<:half_quant} <: Lux.AbstractLuxLayer
    depth::Any
    Φ_fcns::Any
    layernorms::Any
    layernorm_bool::Any
    batchnorm_bool::Any
    x_shape::Any
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
    first(widths) !== q_size && (error(
        "First expert Φ_hidden_widths must be equal to the hidden dimension of the prior.",
        widths,
        " != ",
        q_size,
    ))

    spline_degree = parse(Int, retrieve(conf, "KAN", "spline_degree"))
    layernorm_bool = parse(Bool, retrieve(conf, "GeneratorModel", "layer_norm"))
    base_activation = retrieve(conf, "GeneratorModel", "base_activation")
    spline_function = retrieve(conf, "GeneratorModel", "spline_function")
    grid_size = parse(Int, retrieve(conf, "GeneratorModel", "grid_size"))
    grid_update_ratio =
        parse(half_quant, retrieve(conf, "GeneratorModel", "grid_update_ratio"))
    grid_range = parse.(half_quant, retrieve(conf, "GeneratorModel", "grid_range"))
    ε_scale = parse(half_quant, retrieve(conf, "GeneratorModel", "ε_scale"))
    σ_base = parse(full_quant, retrieve(conf, "GeneratorModel", "σ_base"))
    σ_spline = parse(full_quant, retrieve(conf, "GeneratorModel", "σ_spline"))
    init_τ = parse(full_quant, retrieve(conf, "GeneratorModel", "init_τ"))
    τ_trainable = parse(Bool, retrieve(conf, "GeneratorModel", "τ_trainable"))
    τ_trainable = spline_function == "B-spline" ? false : τ_trainable

    depth = length(widths)-1

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
        )
    Φ_functions = []
    layernorms = []

    for i in eachindex(widths[1:(end-1)])
        base_scale = (
            μ_scale * (one(full_quant) / √(full_quant(widths[i]))) .+
            σ_base .* (
                randn(rng, full_quant, widths[i], widths[i+1]) .* full_quant(2) .-
                one(full_quant)
            ) .* (one(full_quant) / √(full_quant(widths[i])))
        )
        push!(Φ_functions, initialize_function(widths[i], widths[i+1], base_scale))

        if (layernorm_bool && i < depth)
            push!(layernorms, Lux.LayerNorm(widths[i+1]))
        end
    end

    return KAN_Generator(depth, Φ_functions, layernorms, layernorm_bool, false, x_shape)
end

function (gen::KAN_Generator)(
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lyrnorm::NamedTuple,
    z::AbstractArray{T},
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
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
    num_samples = size(z)[end]
    z = dropdims(sum(z, dims = 2), dims = 2)

    # KAN functions
    for i = 1:gen.depth
        z = gen.Φ_fcns[i](z, ps.fcn[symbol_map[i]], st_kan[symbol_map[i]])
        z = dropdims(sum(z, dims = 1); dims = 1)

        z, st_lyrnorm_new =
            (gen.layernorm_bool && i < gen.depth) ?
            Lux.apply(
                gen.layernorms[i],
                z,
                ps.layernorm[symbol_map[i]],
                st_lyrnorm[symbol_map[i]],
            ) : (z, st_lyrnorm)
        (gen.layernorm_bool && i < gen.depth) &&
            @reset st_lyrnorm[symbol_map[i]] = st_lyrnorm_new
    end

    return reshape(z, gen.x_shape..., num_samples), st_lyrnorm
end

end
