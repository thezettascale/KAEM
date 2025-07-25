module CNN_Model

export CNN_Generator, init_CNN_Generator

using CUDA, Lux, LuxCUDA, ComponentArrays, Accessors, Random, ConfParser

using ..Utils

struct CNN_Generator <: Lux.AbstractLuxLayer
    depth::Int
    Φ_fcns::Tuple{Vararg{Lux.ConvTranspose}}
    batchnorms::Tuple{Vararg{Lux.BatchNorm}}
    batchnorm_bool::Bool
    layernorm_bool::Bool
end

function init_CNN_Generator(
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

    widths = (widths..., last(x_shape))

    first(widths) !== q_size && (error(
        "First expert Φ_hidden_widths must be equal to the hidden dimension of the prior.",
        widths,
        " != ",
        q_size,
    ))

    channels = parse.(Int, retrieve(conf, "CNN", "hidden_feature_dims"))
    hidden_c = (q_size, channels...)
    depth = length(hidden_c)-1
    strides = parse.(Int, retrieve(conf, "CNN", "strides"))
    k_size = parse.(Int, retrieve(conf, "CNN", "kernel_sizes"))
    paddings = parse.(Int, retrieve(conf, "CNN", "paddings"))
    act = activation_mapping[retrieve(conf, "CNN", "activation")]
    batchnorm_bool = parse(Bool, retrieve(conf, "CNN", "batchnorm"))

    Φ_functions = Vector{Lux.ConvTranspose}(undef, 0)
    batchnorms = Vector{Lux.BatchNorm}(undef, 0)

    length(strides) != length(hidden_c) &&
        (error("Number of strides must be equal to the number of hidden layers + 1."))
    length(k_size) != length(hidden_c) &&
        (error("Number of kernel sizes must be equal to the number of hidden layers + 1."))
    length(paddings) != length(hidden_c) &&
        (error("Number of paddings must be equal to the number of hidden layers + 1."))

    for i in eachindex(hidden_c[1:(end-1)])
        push!(
            Φ_functions,
            Lux.ConvTranspose(
                (k_size[i], k_size[i]),
                hidden_c[i] => hidden_c[i+1],
                identity;
                stride = strides[i],
                pad = paddings[i],
            ),
        )
        if batchnorm_bool
            push!(batchnorms_temp, Lux.BatchNorm(hidden_c[i+1], act))
        end
    end
    push!(
        Φ_functions,
        Lux.ConvTranspose(
            (k_size[end], k_size[end]),
            hidden_c[end] => last(x_shape),
            identity;
            stride = strides[end],
            pad = paddings[end],
        ),
    )

    depth = length(Φ_functions)

    if length(batchnorms) == 0
        batchnorms = (Lux.BatchNorm(0),)
    end

    return CNN_Generator(depth, (Φ_functions...,), (batchnorms...,), batchnorm_bool, false)
end

function (gen::CNN_Generator)(
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    z::AbstractArray{T},
)::Tuple{AbstractArray{T},NamedTuple} where {T<:half_quant}
    """
    Generate data from the CNN likelihood model.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        rng: The random number generator.
    Returns:
        The generated data.
    """
    z = reshape(sum(z, dims = 2), 1, 1, first(size(z)), last(size(z)))

    for i = 1:(gen.depth)
        z, st_new =
            Lux.apply(gen.Φ_fcns[i], z, ps.fcn[symbol_map[i]], st_lux.fcn[symbol_map[i]])
        @reset st_lux.fcn[symbol_map[i]] = st_new

        z, st_new =
            (gen.batchnorm_bool && i < gen.depth) ?
            Lux.apply(
                gen.batchnorms[i],
                z,
                ps.batchnorm[symbol_map[i]],
                st_lux.batchnorm[symbol_map[i]],
            ) : (z, st_lux)
        (gen.batchnorm_bool && i < gen.depth) &&
            (gen.batchnorm_bool && i < gen.depth) &&
            @reset st_lux.batchnorm[symbol_map[i]] = st_new
    end

    return z, st_lux
end

end
