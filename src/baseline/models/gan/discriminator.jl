module DiscriminatorGAN

export Discriminator, init_discriminator

using Lux, Random, Accessors, NNlib

using ..Utils

struct Discriminator <: Lux.AbstractLuxLayer
    depth::Int
    conv_layers::Tuple{Vararg{Lux.Conv}}
    batchnorms::Tuple{Vararg{Lux.BatchNorm}}
    output_head::Lux.Dense
    bool_config
end

function init_discriminator(
        x_shape::Tuple{Vararg{Int}},
        disc_channels::Vector{Int},
        strides::Vector{Int},
        kernels::Vector{Int},
        paddings::Vector{Int},
        bool_config,
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
        if bool_config.batchnorm && i > 1  # No batchnorm on first layer
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
        bool_config,
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

function (disc::Discriminator)(x, ps, st)
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
