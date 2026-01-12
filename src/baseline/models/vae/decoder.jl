module Decoder

export VAEDecoder, init_decoder

using LUx, NNlib, Accessors, Random

struct VAEDecoder <: Lux.AbstractLuxLayer
    depth::Int
    project::Lux.Dense
    conv_layers::Tuple{Vararg{Lux.ConvTranspose}}
    batchnorms::Tuple{Vararg{Lux.BatchNorm}}
    bool_config
    init_spatial::Int
    init_channels::Int
end

function init_decoder(
        in_channels::Int,
        dec_channels::Vector{Int},
        latent_dim::Int,
        strides::Vector{Int},
        kernels::Vector{Int},
        paddings::Vector{Int},
        bool_config,
        init_spatial::Int,
        init_channels::Int,
    )
    dec_conv_layers = Vector{Lux.ConvTranspose}()
    dec_batchnorms = Vector{Lux.BatchNorm}()

    project = Lux.Dense(latent_dim, init_channels * init_spatial * init_spatial, NNlib.relu)

    prev_c = init_channels
    for (i, c) in enumerate(dec_channels)
        is_last = i == length(dec_channels)
        out_c = is_last ? in_channels : c
        push!(
            dec_conv_layers,
            Lux.ConvTranspose(
                (kernels[i], kernels[i]),
                prev_c => out_c;
                stride = strides[i],
                pad = paddings[i],
            ),
        )
        if bool_config.batchnorm && !is_last
            push!(dec_batchnorms, Lux.BatchNorm(out_c, NNlib.relu))
        end
        prev_c = c
    end

    return VAEDecoder(
        length(dec_conv_layers),
        project,
        Tuple(dec_conv_layers),
        Tuple(dec_batchnorms),
        bool_config,
        init_spatial,
        init_channels,
    )
end

function Lux.initialparameters(rng::AbstractRNG, dec::VAEDecoder)
    dec_conv_ps = NamedTuple(
        symbol_map[i] => Lux.initialparameters(rng, dec.conv_layers[i])
            for i in 1:dec.depth
    )
    dec_bn_ps = dec.bool_config.batchnorm && length(dec.batchnorms) > 0 ? NamedTuple(
            symbol_map[i] => Lux.initialparameters(rng, dec.batchnorms[i])
            for i in 1:length(dec.batchnorms)
        ) : EMPTY_PARAMS

    return (
        project = Lux.initialparameters(rng, dec.project),
        conv = dec_conv_ps,
        batchnorm = dec_bn_ps,
    )
end

function Lux.initialstates(rng::AbstractRNG, dec::VAEDecoder)
    dec_conv_st = NamedTuple(
        symbol_map[i] => Lux.initialstates(rng, dec.conv_layers[i]) |> Lux.f32
            for i in 1:dec.depth
    )
    dec_bn_st = dec.bool_config.batchnorm && length(dec.batchnorms) > 0 ? NamedTuple(
            symbol_map[i] => Lux.initialstates(rng, dec.batchnorms[i]) |> Lux.f32
            for i in 1:length(dec.batchnorms)
        ) : EMPTY_PARAMS

    return (
        project = Lux.initialstates(rng, dec.project) |> Lux.f32,
        conv = dec_conv_st,
        batchnorm = dec_bn_st,
    )
end

function (dec::VAEDecoder)(z, ps, st)
    h, st_proj = Lux.apply(dec.project, z, ps.project, st.project)
    st_new = st
    @reset st_new.project = st_proj

    h = reshape(h, dec.init_spatial, dec.init_spatial, dec.init_channels, size(z, 2))

    for i in 1:dec.depth
        h, st_layer = Lux.apply(
            dec.conv_layers[i], h, ps.conv[symbol_map[i]], st.conv[symbol_map[i]]
        )
        @reset st_new.conv[symbol_map[i]] = st_layer

        is_last = i == dec.depth
        if dec.bool_config.batchnorm && !is_last
            h, st_bn = Lux.apply(
                dec.batchnorms[i], h, ps.batchnorm[symbol_map[i]], st.batchnorm[symbol_map[i]]
            )
            @reset st_new.batchnorm[symbol_map[i]] = st_bn
        elseif !is_last
            h = NNlib.relu(h)
        else
            h = NNlib.sigmoid(h)
        end
    end

    return h, st_new
end

end
