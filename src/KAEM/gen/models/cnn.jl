module CNN_Model

export CNN_Generator, init_CNN_Generator

using CUDA, Lux, LuxCUDA, ComponentArrays, Accessors, Random, ConfParser
using ChainRules.ChainRulesCore: @ignore_derivatives

using ..Utils

struct BoolConfig <: AbstractBoolConfig
    layernorm::Bool
    batchnorm::Bool
    skip_bool::Bool
end

struct CNN_Generator <: Lux.AbstractLuxLayer
    depth::Int
    Φ_fcns::Vector{Lux.ConvTranspose}
    batchnorms::Vector{Lux.BatchNorm}
    bool_config::BoolConfig
end

function upsample_to_match(
    input_tensor::AbstractArray{T,4},
    target_tensor::AbstractArray{T,4},
)::AbstractArray{T,4} where {T<:half_quant}
    input_h, input_w = size(input_tensor, 1), size(input_tensor, 2)
    target_h, target_w = size(target_tensor, 1), size(target_tensor, 2)
    h_factor = div(target_h, input_h)
    w_factor = div(target_w, input_w)
    upsampled = repeat(input_tensor, h_factor, w_factor, 1, 1)
    return upsampled
end

function forward_with_latent_concat(
    gen::CNN_Generator,
    z::AbstractArray{T,4},
    ps::ComponentArray{T},
    st_lux::NamedTuple,
)::Tuple{AbstractArray{T,4},NamedTuple} where {T<:half_quant}

    original_z = z
    current_z = z .* one(T)

    for i = 1:(gen.depth)
        if i > 1
            upsampled_z = upsample_to_match(original_z .* one(T), current_z .* one(T))
            current_z = cat(current_z, upsampled_z, dims = 3)
        end

        current_z, st_new = Lux.apply(
            gen.Φ_fcns[i],
            current_z,
            ps.fcn[symbol_map[i]],
            st_lux.fcn[symbol_map[i]],
        )
        @ignore_derivatives @reset st_lux.fcn[symbol_map[i]] = st_new

        current_z, st_new =
            (gen.bool_config.batchnorm && i < gen.depth) ?
            Lux.apply(
                gen.batchnorms[i],
                current_z,
                ps.batchnorm[symbol_map[i]],
                st_lux.batchnorm[symbol_map[i]],
            ) : (current_z, nothing)
        (gen.bool_config.batchnorm && i < gen.depth) &&
            @ignore_derivatives @reset st_lux.batchnorm[symbol_map[i]] = st_new
    end

    return current_z, st_lux
end

function forward(
    gen::CNN_Generator,
    z::AbstractArray{T,4},
    ps::ComponentArray{T},
    st_lux::NamedTuple,
    current_layer::Int = 1,
    skip_input::Union{AbstractArray{T,4},Nothing} = nothing,
)::Tuple{AbstractArray{T,4},NamedTuple} where {T<:half_quant}
    for i = 1:(gen.depth)
        z, st_new =
            Lux.apply(gen.Φ_fcns[i], z, ps.fcn[symbol_map[i]], st_lux.fcn[symbol_map[i]])
        @ignore_derivatives @reset st_lux.fcn[symbol_map[i]] = st_new

        z, st_new =
            (gen.bool_config.batchnorm && i < gen.depth) ?
            Lux.apply(
                gen.batchnorms[i],
                z,
                ps.batchnorm[symbol_map[i]],
                st_lux.batchnorm[symbol_map[i]],
            ) : (z, nothing)
        (gen.bool_config.batchnorm && i < gen.depth) &&
            (gen.bool_config.batchnorm && i < gen.depth) &&
            @ignore_derivatives @reset st_lux.batchnorm[symbol_map[i]] = st_new
    end

    return z, st_lux
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
    skip_bool = parse(Bool, retrieve(conf, "CNN", "latent_concat")) # Residual connection

    Φ_functions = Vector{Lux.ConvTranspose}(undef, 0)
    batchnorms = Vector{Lux.BatchNorm}(undef, 0)

    length(strides) != length(hidden_c) &&
        (error("Number of strides must be equal to the number of hidden layers + 1."))
    length(k_size) != length(hidden_c) &&
        (error("Number of kernel sizes must be equal to the number of hidden layers + 1."))
    length(paddings) != length(hidden_c) &&
        (error("Number of paddings must be equal to the number of hidden layers + 1."))

    prev_c = 0
    for i in eachindex(hidden_c[1:(end-1)])
        push!(
            Φ_functions,
            Lux.ConvTranspose(
                (k_size[i], k_size[i]),
                hidden_c[i] + prev_c => hidden_c[i+1],
                identity;
                stride = strides[i],
                pad = paddings[i],
            ),
        )

        if batchnorm_bool
            push!(batchnorms_temp, Lux.BatchNorm(hidden_c[i+1], act))
        end

        prev_c = (i == 1 && skip_bool) ? hidden_c[1] : prev_c
    end
    push!(
        Φ_functions,
        Lux.ConvTranspose(
            (k_size[end], k_size[end]),
            hidden_c[end] + prev_c => last(x_shape),
            identity;
            stride = strides[end],
            pad = paddings[end],
        ),
    )

    depth = length(Φ_functions)

    return CNN_Generator(
        depth,
        Φ_functions,
        batchnorms,
        BoolConfig(false, batchnorm_bool, skip_bool),
    )
end

function (gen::CNN_Generator)(
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple,
    z::AbstractArray{T,3},
)::Tuple{AbstractArray{T,4},NamedTuple} where {T<:half_quant}
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
    gen.bool_config.skip_bool && return forward_with_latent_concat(gen, z, ps, st_lux)
    return forward(gen, z, ps, st_lux)
end


end
