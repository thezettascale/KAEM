module GANArchitecture

export Generator, Discriminator, GANConfig, generate, discriminate

using Lux, NNlib, Accessors, Random

using ..Utils

struct GANConfig <: AbstractBoolConfig
    batchnorm::Bool
    spectral_norm::Bool
end

#= Generator =#

struct Generator <: Lux.AbstractLuxLayer
    depth::Int
    project::Lux.Dense
    conv_layers::Tuple{Vararg{Lux.ConvTranspose}}
    batchnorms::Tuple{Vararg{Lux.BatchNorm}}
    bool_config::GANConfig
    init_spatial::Int
    init_channels::Int
end

function init_generator(
        x_shape::Tuple{Vararg{Int}},
        latent_dim::Int,
        gen_channels::Vector{Int},
        strides::Vector{Int},
        kernels::Vector{Int},
        paddings::Vector{Int},
        batchnorm::Bool,
    )
    in_channels = last(x_shape)
    img_size = first(x_shape)

    # Calculate init_spatial based on strides (count stride=2 layers)
    num_upsample = count(s -> s == 2, strides)
    init_spatial = img_size ÷ (2^num_upsample)
    init_channels = first(gen_channels)

    project = Lux.Dense(
        latent_dim,
        init_channels * init_spatial * init_spatial,
        NNlib.relu
    )

    gen_conv_layers = Vector{Lux.ConvTranspose}()
    gen_batchnorms = Vector{Lux.BatchNorm}()

    prev_c = init_channels
    num_layers = length(strides)  # strides array includes final layer

    for i in 1:num_layers
        is_last = i == num_layers
        # For channel layers, use gen_channels; final layer outputs to in_channels
        if i <= length(gen_channels)
            out_c = gen_channels[i]
        else
            out_c = in_channels
        end
        if is_last
            out_c = in_channels
        end

        push!(
            gen_conv_layers,
            Lux.ConvTranspose(
                (kernels[i], kernels[i]),
                prev_c => out_c;
                stride = strides[i],
                pad = paddings[i],
            ),
        )

        if batchnorm && !is_last
            push!(gen_batchnorms, Lux.BatchNorm(out_c, NNlib.relu))
        end
        prev_c = out_c
    end

    return Generator(
        length(gen_conv_layers),
        project,
        Tuple(gen_conv_layers),
        Tuple(gen_batchnorms),
        GANConfig(batchnorm, false),
        init_spatial,
        init_channels,
    )
end

function Lux.initialparameters(rng::AbstractRNG, gen::Generator)
    gen_conv_ps = NamedTuple(
        symbol_map[i] => Lux.initialparameters(rng, gen.conv_layers[i])
            for i in 1:gen.depth
    )
    gen_bn_ps = gen.bool_config.batchnorm && length(gen.batchnorms) > 0 ? NamedTuple(
            symbol_map[i] => Lux.initialparameters(rng, gen.batchnorms[i])
            for i in 1:length(gen.batchnorms)
        ) : EMPTY_PARAMS

    return (
        project = Lux.initialparameters(rng, gen.project),
        conv = gen_conv_ps,
        batchnorm = gen_bn_ps,
    )
end

function Lux.initialstates(rng::AbstractRNG, gen::Generator)
    gen_conv_st = NamedTuple(
        symbol_map[i] => Lux.initialstates(rng, gen.conv_layers[i]) |> Lux.f32
            for i in 1:gen.depth
    )
    gen_bn_st = gen.bool_config.batchnorm && length(gen.batchnorms) > 0 ? NamedTuple(
            symbol_map[i] => Lux.initialstates(rng, gen.batchnorms[i]) |> Lux.f32
            for i in 1:length(gen.batchnorms)
        ) : EMPTY_PARAMS

    return (
        project = Lux.initialstates(rng, gen.project) |> Lux.f32,
        conv = gen_conv_st,
        batchnorm = gen_bn_st,
    )
end

function generate(gen::Generator, z, ps, st)
    h, st_proj = Lux.apply(gen.project, z, ps.project, st.project)
    st_new = st
    @reset st_new.project = st_proj

    h = reshape(h, gen.init_spatial, gen.init_spatial, gen.init_channels, size(z, 2))

    bn_idx = 1
    for i in 1:gen.depth
        h, st_layer = Lux.apply(
            gen.conv_layers[i], h, ps.conv[symbol_map[i]], st.conv[symbol_map[i]]
        )
        @reset st_new.conv[symbol_map[i]] = st_layer

        is_last = i == gen.depth
        if gen.bool_config.batchnorm && bn_idx <= length(gen.batchnorms) && !is_last
            h, st_bn = Lux.apply(
                gen.batchnorms[bn_idx], h,
                ps.batchnorm[symbol_map[bn_idx]], st.batchnorm[symbol_map[bn_idx]]
            )
            @reset st_new.batchnorm[symbol_map[bn_idx]] = st_bn
            bn_idx += 1
        elseif !is_last
            h = NNlib.relu(h)
        else
            h = (NNlib.tanh_fast(h) .+ 1.0f0) ./ 2.0f0
        end
    end

    return h, st_new
end

#= Discriminator =#

struct Discriminator <: Lux.AbstractLuxLayer
    depth::Int
    conv_layers::Tuple{Vararg{Lux.Conv}}
    batchnorms::Tuple{Vararg{Lux.BatchNorm}}
    output_head::Lux.Dense
    bool_config::GANConfig
end

function init_discriminator(
        x_shape::Tuple{Vararg{Int}},
        disc_channels::Vector{Int},
        strides::Vector{Int},
        kernels::Vector{Int},
        paddings::Vector{Int},
        batchnorm::Bool,
    )
    in_channels = last(x_shape)
    img_size = first(x_shape)

    disc_conv_layers = Vector{Lux.Conv}()
    disc_batchnorms = Vector{Lux.BatchNorm}()

    prev_c = in_channels
    spatial = img_size
    for (i, c) in enumerate(disc_channels)
        push!(
            disc_conv_layers,
            Lux.Conv(
                (kernels[i], kernels[i]),
                prev_c => c;
                stride = strides[i],
                pad = paddings[i],
            ),
        )
        if batchnorm && i > 1  # No batchnorm on first layer
            push!(disc_batchnorms, Lux.BatchNorm(c, NNlib.leakyrelu))
        end
        prev_c = c
        spatial = div(spatial - kernels[i] + 2 * paddings[i], strides[i]) + 1
    end

    output_dim = prev_c * spatial * spatial
    output_head = Lux.Dense(output_dim, 1)

    return Discriminator(
        length(disc_conv_layers),
        Tuple(disc_conv_layers),
        Tuple(disc_batchnorms),
        output_head,
        GANConfig(batchnorm, false),
    )
end

function Lux.initialparameters(rng::AbstractRNG, disc::Discriminator)
    disc_conv_ps = NamedTuple(
        symbol_map[i] => Lux.initialparameters(rng, disc.conv_layers[i])
            for i in 1:disc.depth
    )
    disc_bn_ps = disc.bool_config.batchnorm && length(disc.batchnorms) > 0 ? NamedTuple(
            symbol_map[i] => Lux.initialparameters(rng, disc.batchnorms[i])
            for i in 1:length(disc.batchnorms)
        ) : EMPTY_PARAMS

    return (
        conv = disc_conv_ps,
        batchnorm = disc_bn_ps,
        output = Lux.initialparameters(rng, disc.output_head),
    )
end

function Lux.initialstates(rng::AbstractRNG, disc::Discriminator)
    disc_conv_st = NamedTuple(
        symbol_map[i] => Lux.initialstates(rng, disc.conv_layers[i]) |> Lux.f32
            for i in 1:disc.depth
    )
    disc_bn_st = disc.bool_config.batchnorm && length(disc.batchnorms) > 0 ? NamedTuple(
            symbol_map[i] => Lux.initialstates(rng, disc.batchnorms[i]) |> Lux.f32
            for i in 1:length(disc.batchnorms)
        ) : EMPTY_PARAMS

    return (
        conv = disc_conv_st,
        batchnorm = disc_bn_st,
        output = Lux.initialstates(rng, disc.output_head) |> Lux.f32,
    )
end

function discriminate(disc::Discriminator, x, ps, st)
    h = x
    st_new = st

    bn_idx = 1
    for i in 1:disc.depth
        h, st_layer = Lux.apply(
            disc.conv_layers[i], h, ps.conv[symbol_map[i]], st.conv[symbol_map[i]]
        )
        @reset st_new.conv[symbol_map[i]] = st_layer

        if disc.bool_config.batchnorm && i > 1 && bn_idx <= length(disc.batchnorms)
            h, st_bn = Lux.apply(
                disc.batchnorms[bn_idx], h,
                ps.batchnorm[symbol_map[bn_idx]], st.batchnorm[symbol_map[bn_idx]]
            )
            @reset st_new.batchnorm[symbol_map[bn_idx]] = st_bn
            bn_idx += 1
        else
            h = NNlib.leakyrelu(h)
        end
    end

    h_flat = reshape(h, :, size(h, 4))
    logits, st_out = Lux.apply(disc.output_head, h_flat, ps.output, st.output)
    @reset st_new.output = st_out

    return logits, st_new
end

end
