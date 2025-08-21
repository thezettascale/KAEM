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

function apply_skip_connection(
    skip_input::AbstractArray{T,4},
    target::AbstractArray{T,4},
)::AbstractArray{T,4} where {T<:half_quant}
    size(skip_input) == size(target) && return target + skip_input
    upsampled_skip = upsample_to_match(skip_input, target)
    return target + upsampled_skip
end

function layer_with_skip(
    gen::CNN_Generator,
    z::AbstractArray{T,4},
    ps::ComponentArray{T},
    st_lux::NamedTuple,
    layer_idx::Int,
    skip_input::Union{AbstractArray{T,4},Nothing},
)::Tuple{AbstractArray{T,4},NamedTuple} where {T<:half_quant}

    z, st_new = Lux.apply(
        gen.Φ_fcns[layer_idx],
        z,
        ps.fcn[symbol_map[layer_idx]],
        st_lux.fcn[symbol_map[layer_idx]],
    )
    @ignore_derivatives @reset st_lux.fcn[symbol_map[layer_idx]] = st_new

    if gen.bool_config.batchnorm && layer_idx < gen.depth
        z, st_new = Lux.apply(
            gen.batchnorms[layer_idx],
            z,
            ps.batchnorm[symbol_map[layer_idx]],
            st_lux.batchnorm[symbol_map[layer_idx]],
        )
        @ignore_derivatives @reset st_lux.batchnorm[symbol_map[layer_idx]] = st_new
    end

    if gen.bool_config.skip_bool && skip_input !== nothing
        z = apply_skip_connection(skip_input, z)
    end

    return z, st_lux
end

function forward_recursive(
    gen::CNN_Generator,
    z::AbstractArray{T,4},
    ps::ComponentArray{T},
    st_lux::NamedTuple,
    current_layer::Int = 1,
    skip_input::Union{AbstractArray{T,4},Nothing} = nothing,
)::Tuple{AbstractArray{T,4},NamedTuple} where {T<:half_quant}
    current_layer == gen.depth &&
        return layer_with_skip(gen, z, ps, st_lux, current_layer, nothing)
    z_layer, st_lux = layer_with_skip(gen, z, ps, st_lux, current_layer, skip_input)
    next_skip_input = gen.bool_config.skip_bool ? z_layer : nothing
    return forward_recursive(gen, z_layer, ps, st_lux, current_layer + 1, next_skip_input)
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
    skip_bool = parse(Bool, retrieve(conf, "CNN", "residual_connections")) # Residual connection

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
    gen.bool_config.skip_bool && return forward_recursive(gen, z, ps, st_lux)
    return forward(gen, z, ps, st_lux)
end


end
