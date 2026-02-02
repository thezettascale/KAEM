module CNN_Model

export CNN_Generator, init_CNN_Generator

using Lux, ComponentArrays, Accessors, Random, ConfParser

using ..Utils

struct BoolConfig <: AbstractBoolConfig
    layernorm::Bool
    batchnorm::Bool
    skip_bool::Bool
    projection_bool::Bool
end

struct CNN_Generator <: Lux.AbstractLuxLayer
    depth::Int
    Φ_fcns::Tuple{Vararg{Lux.ConvTranspose}}
    batchnorms::Tuple{Vararg{Lux.BatchNorm}}
    project::Union{Lux.Dense, Nothing}
    bool_config::BoolConfig
    in_channels::Int
    s_size::Int
    init_spatial::Int
    init_channels::Int
end

function upsample_to_match(
        input_tensor,
        target_tensor,
    )
    input_h, input_w = size(input_tensor, 1), size(input_tensor, 2)
    target_h, target_w = size(target_tensor, 1), size(target_tensor, 2)
    h_factor = div(target_h, input_h)
    w_factor = div(target_w, input_w)
    upsampled = repeat(input_tensor, h_factor, w_factor, 1, 1)
    return upsampled
end

function forward_with_latent_concat(
        gen::CNN_Generator,
        z,
        ps,
        st_lux,
    )

    z_first, st_layer_new = Lux.apply(
        gen.Φ_fcns[1],
        z,
        ps.fcn[symbol_map[1]],
        st_lux.fcn[symbol_map[1]],
    )
    @reset st_lux.fcn[symbol_map[1]] = st_layer_new

    state = (1, st_lux, z_first)
    while first(state) < gen.depth
        i, st_lux_new, z_acc = state
        if gen.bool_config.batchnorm
            z_acc, st_layer_new = Lux.apply(
                gen.batchnorms[i],
                z_acc,
                ps.batchnorm[symbol_map[i]],
                st_lux_new.batchnorm[symbol_map[i]],
            )
            @reset st_lux_new.batchnorm[symbol_map[i]] = st_layer_new
        end

        i += 1

        upsampled_z = upsample_to_match(z .* 1.0f0, z_acc .* 1.0f0)
        z_i = cat(z_acc, upsampled_z, dims = 3)

        z_i, st_layer_new = Lux.apply(
            gen.Φ_fcns[i],
            z_i,
            ps.fcn[symbol_map[i]],
            st_lux_new.fcn[symbol_map[i]],
        )
        @reset st_lux_new.fcn[symbol_map[i]] = st_layer_new

        state = (i, st_lux_new, z_i)
    end

    _, st_lux_new, z = state
    return z, st_lux_new
end

function forward(
        gen::CNN_Generator,
        z,
        ps,
        st_lux,
        current_layer = 1,
        skip_input = nothing,
    )
    state = (1, st_lux, z .* 1.0f0)
    while first(state) < gen.depth
        i, st_lux_new, z_acc = state
        z_acc, st_layer_new =
            Lux.apply(gen.Φ_fcns[i], z_acc, ps.fcn[symbol_map[i]], st_lux_new.fcn[symbol_map[i]])
        @reset st_lux_new.fcn[symbol_map[i]] = st_layer_new

        if gen.bool_config.batchnorm
            z_acc, st_layer_new = Lux.apply(
                gen.batchnorms[i],
                z_acc,
                ps.batchnorm[symbol_map[i]],
                st_lux_new.batchnorm[symbol_map[i]],
            )

            @reset st_lux_new.fcn[symbol_map[i]] = st_layer_new
        end

        state = (i + 1, st_lux_new, z_acc)
    end

    _, st_lux_new, z = state
    z, st_layer_new =
        Lux.apply(gen.Φ_fcns[gen.depth], z, ps.fcn[symbol_map[gen.depth]], st_lux_new.fcn[symbol_map[gen.depth]])
    @reset st_lux_new.fcn[symbol_map[gen.depth]] = st_layer_new

    return z, st_lux_new
end

function init_CNN_Generator(
        conf::ConfParse,
        x_shape::Tuple,
        rng::AbstractRNG = Random.default_rng(),
    )

    prior_widths = parse_config_array(Int, retrieve(conf, "EbmModel", "layer_widths"))
    q_size = get_q_size(prior_widths)
    widths = parse_config_array(Int, retrieve(conf, "GeneratorModel", "widths"))
    widths = (widths..., last(x_shape))
    validate_generator_widths(widths, q_size)

    channels = parse.(Int, retrieve(conf, "CNN", "hidden_feature_dims"))
    strides = parse.(Int, retrieve(conf, "CNN", "strides"))
    k_size = parse.(Int, retrieve(conf, "CNN", "kernel_sizes"))
    paddings = parse.(Int, retrieve(conf, "CNN", "paddings"))
    act = lux_activation_mapping[retrieve(conf, "CNN", "activation")]
    batchnorm_bool = parse(Bool, retrieve(conf, "CNN", "batchnorm"))
    skip_bool = parse(Bool, retrieve(conf, "CNN", "latent_concat")) # Residual connection
    projection_bool = parse(Bool, retrieve(conf, "CNN", "projection"))

    # Compute init_spatial and init_channels like VAE/GAN decoders:
    # init_spatial = img_size / (2^num_stride2_layers)
    # init_channels = first(hidden_feature_dims)
    img_size = first(x_shape)
    num_upsample = count(s -> s == 2, strides)
    init_spatial = img_size ÷ (2^num_upsample)
    init_channels = first(channels)

    s_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    # Create projection layer if enabled
    # Projects from q_size to init_channels * init_spatial^2, then reshape to spatial
    project = projection_bool ?
        Lux.Dense(q_size, init_channels * init_spatial * init_spatial, act) :
        nothing

    # Channel configuration depends on whether projection is used
    # With projection: start from init_channels (same as first conv output)
    # Without projection: start from q_size (latent dim) at 1x1 spatial
    first_in_channels = projection_bool ? init_channels : q_size
    hidden_c = (first_in_channels, channels...)
    depth = length(hidden_c) - 1

    Φ_functions = Vector{Lux.ConvTranspose}(undef, 0)
    batchnorms = Vector{Lux.BatchNorm}(undef, 0)

    length(strides) != length(hidden_c) &&
        (error("Number of strides must be equal to the number of hidden layers + 1."))
    length(k_size) != length(hidden_c) &&
        (error("Number of kernel sizes must be equal to the number of hidden layers + 1."))
    length(paddings) != length(hidden_c) &&
        (error("Number of paddings must be equal to the number of hidden layers + 1."))

    prev_c = 0
    for i in eachindex(hidden_c[1:(end - 1)])
        push!(
            Φ_functions,
            Lux.ConvTranspose(
                (k_size[i], k_size[i]),
                hidden_c[i] + prev_c => hidden_c[i + 1],
                identity;
                stride = strides[i],
                pad = paddings[i],
            ),
        )

        if batchnorm_bool
            push!(batchnorms, Lux.BatchNorm(hidden_c[i + 1], act))
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
        Tuple(Φ_functions),
        Tuple(batchnorms),
        project,
        BoolConfig(false, batchnorm_bool, skip_bool, projection_bool),
        first(widths),
        s_size,
        init_spatial,
        init_channels,  # Derived from first(channels) / hidden_feature_dims[1]
    )
end

function (gen::CNN_Generator)(
        ps,
        st_kan,
        st_lux,
        z,
    )
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
    z_summed = sum(z, dims = 2)

    # Projection matches baseline VAE decoder architecture
    if gen.bool_config.projection_bool
        z_flat = dropdims(z_summed, dims = 2)
        z_proj, st_proj_new = Lux.apply(gen.project, z_flat, ps.project, st_lux.project)
        @reset st_lux.project = st_proj_new
        z_spatial = reshape(z_proj, gen.init_spatial, gen.init_spatial, gen.init_channels, gen.s_size)
    else
        z_spatial = reshape(z_summed, 1, 1, gen.in_channels, gen.s_size)
    end

    out = (
        gen.bool_config.skip_bool ?
            forward_with_latent_concat(gen, z_spatial, ps, st_lux) :
            forward(gen, z_spatial, ps, st_lux)
    )
    return out
end


end
