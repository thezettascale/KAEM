module CNN_Encoder_Model

export CNN_Encoder, init_cnn_encoder

using Lux, Random, ConfParser, NNlib, Accessors

using ..Utils

struct BoolConfig <: AbstractBoolConfig
    batchnorm::Bool
end

struct CNN_Encoder <: Lux.AbstractLuxLayer
    depth::Int
    conv_layers::Tuple{Vararg{Lux.Conv}}
    batchnorms::Tuple{Vararg{Lux.BatchNorm}}
    flatten_dense::Lux.Dense
    mu_head::Lux.Dense
    logvar_head::Lux.Dense
    bool_config::BoolConfig
    latent_dim::Int
    input_shape::Tuple{Vararg{Int}}
    flat_size::Int
    s_size::Int
end

function init_cnn_encoder(
        conf::ConfParse,
        x_shape::Tuple{Vararg{Int}},
        rng::AbstractRNG,
    )
    prior_widths = parse_config_array(Int, retrieve(conf, "EbmModel", "layer_widths"))
    q_size = first(prior_widths)
    p_size = last(prior_widths)
    latent_dim = q_size * p_size

    channels = parse_config_array(Int, retrieve(conf, "Encoder", "channels"))
    strides = parse_config_array(Int, retrieve(conf, "Encoder", "strides"))
    k_sizes = parse_config_array(Int, retrieve(conf, "Encoder", "kernel_sizes"))
    paddings = parse_config_array(Int, retrieve(conf, "Encoder", "paddings"))
    batchnorm_bool = parse(Bool, retrieve(conf, "Encoder", "batchnorm"))
    s_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    in_channels = last(x_shape)
    hidden_c = (in_channels, channels...)
    depth = length(channels)

    length(strides) != depth &&
        error("Number of strides must equal number of channel layers: ", length(strides), " != ", depth)
    length(k_sizes) != depth &&
        error("Number of kernel sizes must equal number of channel layers: ", length(k_sizes), " != ", depth)
    length(paddings) != depth &&
        error("Number of paddings must equal number of channel layers: ", length(paddings), " != ", depth)

    conv_layers = Vector{Lux.Conv}(undef, 0)
    batchnorms = Vector{Lux.BatchNorm}(undef, 0)

    for i in 1:depth
        push!(
            conv_layers,
            Lux.Conv(
                (k_sizes[i], k_sizes[i]),
                hidden_c[i] => hidden_c[i + 1],
                identity;
                stride = strides[i],
                pad = paddings[i],
            ),
        )

        if batchnorm_bool
            push!(batchnorms, Lux.BatchNorm(hidden_c[i + 1], NNlib.gelu))
        end
    end

    # Calculate flattened size after convolutions
    h, w = x_shape[1], x_shape[2]
    for i in 1:depth
        h = div(h - k_sizes[i] + 2 * paddings[i], strides[i]) + 1
        w = div(w - k_sizes[i] + 2 * paddings[i], strides[i]) + 1
    end
    flat_size = h * w * last(channels)

    flatten_dense = Lux.Dense(flat_size => latent_dim, NNlib.gelu)
    mu_head = Lux.Dense(latent_dim => latent_dim)
    logvar_head = Lux.Dense(latent_dim => latent_dim)

    return CNN_Encoder(
        depth,
        Tuple(conv_layers),
        Tuple(batchnorms),
        flatten_dense,
        mu_head,
        logvar_head,
        BoolConfig(batchnorm_bool),
        latent_dim,
        x_shape,
        flat_size,
        s_size,
    )
end

function (enc::CNN_Encoder)(
        ps,
        st_lux,
        x,
        ε,
    )
    h = reshape(x, enc.input_shape..., enc.s_size)
    st_new = st_lux

    for i in 1:enc.depth
        h, st_layer = Lux.apply(
            enc.conv_layers[i],
            h,
            ps.conv[symbol_map[i]],
            st_new.conv[symbol_map[i]]
        )
        @reset st_new.conv[symbol_map[i]] = st_layer

        if enc.bool_config.batchnorm
            h, st_bn = Lux.apply(
                enc.batchnorms[i],
                h,
                ps.batchnorm[symbol_map[i]],
                st_new.batchnorm[symbol_map[i]]
            )
            @reset st_new.batchnorm[symbol_map[i]] = st_bn
        else
            h = NNlib.gelu(h)
        end
    end

    h_flat = reshape(h, enc.flat_size, enc.s_size)

    h_dense, st_dense = Lux.apply(
        enc.flatten_dense,
        h_flat,
        ps.flatten,
        st_new.flatten
    )
    @reset st_new.flatten = st_dense

    μ, st_mu = Lux.apply(enc.mu_head, h_dense, ps.mu, st_new.mu)
    @reset st_new.mu = st_mu

    logvar, st_logvar = Lux.apply(
        enc.logvar_head,
        h_dense,
        ps.logvar,
        st_new.logvar
    )
    @reset st_new.logvar = st_logvar

    logvar = clamp.(logvar, -10.0f0, 2.0f0)
    σ = exp.(0.5f0 .* logvar)
    z = μ .+ σ .* ε

    log_q = -0.5f0 .* sum(
        logvar .+ (z .- μ) .^ 2 ./ exp.(logvar) .+ log(2.0f0 * Float32(π));
        dims = 1
    )
    log_q = dropdims(log_q; dims = 1)

    return z, log_q, μ, logvar, st_new
end

end
