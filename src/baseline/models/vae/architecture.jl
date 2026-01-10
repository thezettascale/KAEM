export VAEEncoder, VAEDecoder, VAEConfig, encode, decode

struct VAEConfig <: AbstractBoolConfig
    batchnorm::Bool
end

#= Encoder =#

struct VAEEncoder <: Lux.AbstractLuxLayer
    depth::Int
    conv_layers::Tuple{Vararg{Lux.Conv}}
    batchnorms::Tuple{Vararg{Lux.BatchNorm}}
    flatten_dense::Lux.Dense
    mu_head::Lux.Dense
    logvar_head::Lux.Dense
    bool_config::VAEConfig
    latent_dim::Int
end

function init_encoder(
        x_shape::Tuple{Vararg{Int}},
        enc_channels::Vector{Int},
        latent_dim::Int,
        kernel_size::Int,
        batchnorm::Bool,
    )
    in_channels = last(x_shape)
    img_size = first(x_shape)

    enc_conv_layers = Vector{Lux.Conv}()
    enc_batchnorms = Vector{Lux.BatchNorm}()

    prev_c = in_channels
    spatial = img_size
    for c in enc_channels
        push!(
            enc_conv_layers,
            Lux.Conv(
                (kernel_size, kernel_size),
                prev_c => c,
                NNlib.leakyrelu;
                stride = 2,
                pad = 1,
            ),
        )
        if batchnorm
            push!(enc_batchnorms, Lux.BatchNorm(c))
        end
        prev_c = c
        spatial = div(spatial, 2)
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
        VAEConfig(batchnorm),
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

function encode(enc::VAEEncoder, x, ps, st)
    h = x
    st_new = st

    for i in 1:enc.depth
        h, st_layer = Lux.apply(
            enc.conv_layers[i], h, ps.conv[symbol_map[i]], st.conv[symbol_map[i]]
        )
        @reset st_new.conv[symbol_map[i]] = st_layer

        if enc.bool_config.batchnorm
            h, st_bn = Lux.apply(
                enc.batchnorms[i], h, ps.batchnorm[symbol_map[i]], st.batchnorm[symbol_map[i]]
            )
            @reset st_new.batchnorm[symbol_map[i]] = st_bn
        end
    end

    h_flat = reshape(h, :, size(h, 4))
    h_dense, st_flat = Lux.apply(enc.flatten_dense, h_flat, ps.flatten, st.flatten)
    @reset st_new.flatten = st_flat

    μ, st_mu = Lux.apply(enc.mu_head, h_dense, ps.mu, st.mu)
    @reset st_new.mu = st_mu

    logvar, st_logvar = Lux.apply(enc.logvar_head, h_dense, ps.logvar, st.logvar)
    @reset st_new.logvar = st_logvar

    return μ, logvar, st_new
end

#= Decoder =#

struct VAEDecoder <: Lux.AbstractLuxLayer
    depth::Int
    project::Lux.Dense
    conv_layers::Tuple{Vararg{Lux.ConvTranspose}}
    batchnorms::Tuple{Vararg{Lux.BatchNorm}}
    bool_config::VAEConfig
    init_spatial::Int
    init_channels::Int
end

function init_decoder(
        in_channels::Int,
        dec_channels::Vector{Int},
        latent_dim::Int,
        kernel_size::Int,
        batchnorm::Bool,
        init_spatial::Int,
        init_channels::Int,
    )
    dec_conv_layers = Vector{Lux.ConvTranspose}()
    dec_batchnorms = Vector{Lux.BatchNorm}()

    project = Lux.Dense(latent_dim, init_channels * init_spatial * init_spatial, NNlib.relu)

    prev_c = init_channels
    for (i, c) in enumerate(dec_channels)
        is_last = i == length(dec_channels)
        act = is_last ? NNlib.sigmoid : NNlib.relu
        out_c = is_last ? in_channels : c
        push!(
            dec_conv_layers,
            Lux.ConvTranspose(
                (kernel_size, kernel_size),
                prev_c => out_c,
                act;
                stride = 2,
                pad = 1,
            ),
        )
        if batchnorm && !is_last
            push!(dec_batchnorms, Lux.BatchNorm(out_c))
        end
        prev_c = c
    end

    return VAEDecoder(
        length(dec_conv_layers),
        project,
        Tuple(dec_conv_layers),
        Tuple(dec_batchnorms),
        VAEConfig(batchnorm),
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

function decode(dec::VAEDecoder, z, ps, st)
    h, st_proj = Lux.apply(dec.project, z, ps.project, st.project)
    st_new = st
    @reset st_new.project = st_proj

    h = reshape(h, dec.init_spatial, dec.init_spatial, dec.init_channels, size(z, 2))

    for i in 1:dec.depth
        h, st_layer = Lux.apply(
            dec.conv_layers[i], h, ps.conv[symbol_map[i]], st.conv[symbol_map[i]]
        )
        @reset st_new.conv[symbol_map[i]] = st_layer

        if dec.bool_config.batchnorm && i < dec.depth
            h, st_bn = Lux.apply(
                dec.batchnorms[i], h, ps.batchnorm[symbol_map[i]], st.batchnorm[symbol_map[i]]
            )
            @reset st_new.batchnorm[symbol_map[i]] = st_bn
        end
    end

    return h, st_new
end
