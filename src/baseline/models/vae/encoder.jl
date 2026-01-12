module Encoder

export VAEEncoder, init_encoder

using Lux, Accessors, NNlib, Random

using ..Utils

struct VAEEncoder <: Lux.AbstractLuxLayer
    depth::Int
    conv_layers::Tuple{Vararg{Lux.Conv}}
    batchnorms::Tuple{Vararg{Lux.BatchNorm}}
    flatten_dense::Lux.Dense
    mu_head::Lux.Dense
    logvar_head::Lux.Dense
    bool_config
    latent_dim::Int
end

function init_encoder(
        x_shape::Tuple{Vararg{Int}},
        enc_channels::Vector{Int},
        latent_dim::Int,
        strides::Vector{Int},
        kernels::Vector{Int},
        paddings::Vector{Int},
        bool_config,
    )
    in_channels = last(x_shape)
    img_size = first(x_shape)

    enc_conv_layers = Vector{Lux.Conv}()
    enc_batchnorms = Vector{Lux.BatchNorm}()

    prev_c = in_channels
    spatial = img_size
    for (i, c) in enumerate(enc_channels)
        push!(
            enc_conv_layers,
            Lux.Conv(
                (kernels[i], kernels[i]),
                prev_c => c;
                stride = strides[i],
                pad = paddings[i],
            ),
        )
        if bool_config.batchnorm
            push!(enc_batchnorms, Lux.BatchNorm(c, NNlib.leakyrelu))
        end
        prev_c = c
        spatial = div(spatial - kernels[i] + 2 * paddings[i], strides[i]) + 1
    end

    flatten_dim = prev_c * spatial * spatial
    flatten_dense = Lux.Dense(flatten_dim, 256, NNlib.relu)
    mu_head = Lux.Dense(256, latent_dim)
    logvar_head = Lux.Dense(256, latent_dim)

    encoder = VAEEncoder(
        length(enc_channels),
        Tuple(enc_conv_layers),
        Tuple(enc_batchnorms),
        flatten_dense,
        mu_head,
        logvar_head,
        bool_config,
        latent_dim,
    )

    return encoder, spatial, prev_c
end

function Lux.initialparameters(rng::AbstractRNG, enc::VAEEncoder)
    enc_conv_ps = NamedTuple(
        symbol_map[i] => Lux.initialparameters(rng, enc.conv_layers[i])
            for i in 1:enc.depth
    )
    enc_bn_ps = enc.bool_config.batchnorm ? NamedTuple(
            symbol_map[i] => Lux.initialparameters(rng, enc.batchnorms[i])
            for i in 1:length(enc.batchnorms)
        ) : EMPTY_PARAMS

    return (
        conv = enc_conv_ps,
        batchnorm = enc_bn_ps,
        flatten = Lux.initialparameters(rng, enc.flatten_dense),
        mu = Lux.initialparameters(rng, enc.mu_head),
        logvar = Lux.initialparameters(rng, enc.logvar_head),
    )
end

function Lux.initialstates(rng::AbstractRNG, enc::VAEEncoder)
    enc_conv_st = NamedTuple(
        symbol_map[i] => Lux.initialstates(rng, enc.conv_layers[i]) |> Lux.f32
            for i in 1:enc.depth
    )
    enc_bn_st = enc.bool_config.batchnorm ? NamedTuple(
            symbol_map[i] => Lux.initialstates(rng, enc.batchnorms[i]) |> Lux.f32
            for i in 1:length(enc.batchnorms)
        ) : EMPTY_PARAMS

    return (
        conv = enc_conv_st,
        batchnorm = enc_bn_st,
        flatten = Lux.initialstates(rng, enc.flatten_dense) |> Lux.f32,
        mu = Lux.initialstates(rng, enc.mu_head) |> Lux.f32,
        logvar = Lux.initialstates(rng, enc.logvar_head) |> Lux.f32,
    )
end

function (enc::VAEEncoder)(x, ps, st)
    h = x
    st_new = st

    for i in 1:enc.depth
        h, st_layer = Lux.apply(
            enc.conv_layers[i], h, ps.conv[symbol_map[i]], st.conv[symbol_map[i]]
        )
        @reset st_new.conv[symbol_map[i]] = st_layer

        if enc.bool_config.batchnorm
            h, st_bn = Lux.apply(
                enc.batchnorms[i],
                h,
                ps.batchnorm[symbol_map[i]],
                st.batchnorm[symbol_map[i]]
            )
            @reset st_new.batchnorm[symbol_map[i]] = st_bn
        else
            h = NNlib.leakyrelu(h)
        end
    end

    h_flat = reshape(h, :, size(h, 4))
    h_dense, st_flat = Lux.apply(
        enc.flatten_dense,
        h_flat,
        ps.flatten,
        st.flatten
    )
    @reset st_new.flatten = st_flat

    μ, st_mu = Lux.apply(enc.mu_head, h_dense, ps.mu, st.mu)
    @reset st_new.mu = st_mu

    logvar, st_logvar = Lux.apply(
        enc.logvar_head,
        h_dense,
        ps.logvar,
        st.logvar
    )
    @reset st_new.logvar = st_logvar

    return μ, logvar, st_new
end

end
