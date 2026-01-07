module Diagonal_Model

export DiagonalGaussianEncoder, init_diagonal_encoder

using Lux, Random, ConfParser, NNlib, Accessors

using ..Utils

struct DiagonalGaussianEncoder <: Lux.AbstractLuxLayer
    depth::Int
    layers::Tuple{Vararg{Lux.Dense}}
    mu_head::Lux.Dense
    logvar_head::Lux.Dense
    latent_dim::Int
    input_dim::Int
    s_size::Int
end

function init_diagonal_encoder(
        conf::ConfParse,
        x_shape::Tuple{Vararg{Int}},
        rng::AbstractRNG,
    )
    prior_widths = parse_config_array(Int, retrieve(conf, "EbmModel", "layer_widths"))
    q_size = first(prior_widths)
    p_size = last(prior_widths)
    latent_dim = q_size * p_size

    encoder_widths = parse_config_array(Int, retrieve(conf, "Encoder", "widths"))
    s_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    input_dim = prod(x_shape)
    layers = Vector{Lux.Dense}(undef, 0)

    prev_dim = input_dim
    for width in encoder_widths
        push!(layers, Lux.Dense(prev_dim => width, NNlib.gelu))
        prev_dim = width
    end

    mu_head = Lux.Dense(prev_dim => latent_dim)
    logvar_head = Lux.Dense(prev_dim => latent_dim)

    return DiagonalGaussianEncoder(
        length(layers),
        Tuple(layers),
        mu_head,
        logvar_head,
        latent_dim,
        input_dim,
        s_size,
    )
end

function (enc::DiagonalGaussianEncoder)(
        ps,
        st_lux,
        x,
        ε,
    )
    h = reshape(x, enc.input_dim, enc.s_size)
    st_new = st_lux

    for i in 1:enc.depth
        h, st_layer = Lux.apply(
            enc.layers[i],
            h,
            ps.layers[symbol_map[i]],
            st_new.layers[symbol_map[i]]
        )
        @reset st_new.layers[symbol_map[i]] = st_layer
    end

    μ, st_mu = Lux.apply(enc.mu_head, h, ps.mu, st_new.mu)
    @reset st_new.mu = st_mu

    logvar, st_logvar = Lux.apply(
        enc.logvar_head,
        h,
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
