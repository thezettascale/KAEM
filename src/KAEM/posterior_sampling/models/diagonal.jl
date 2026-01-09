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
    q_size::Int
    p_size::Int
    s_size::Int
end

function init_diagonal_encoder(
        conf::ConfParse,
        x_shape::Tuple{Vararg{Int}},
        rng::AbstractRNG,
    )
    prior_widths = parse_config_array(Int, retrieve(conf, "EbmModel", "layer_widths"))
    p_size = first(prior_widths)
    q_size = last(prior_widths)
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
        q_size,
        p_size,
        s_size,
    )
end

function (enc::DiagonalGaussianEncoder)(
        ps,
        st_lux,
        x,
        ε;
        component_mask = nothing,
    )
    Q, P, S = enc.q_size, enc.p_size, enc.s_size
    h = reshape(x, enc.input_dim, S)
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

    μ_flat, st_mu = Lux.apply(enc.mu_head, h, ps.mu, st_new.mu)
    @reset st_new.mu = st_mu

    logvar_flat, st_logvar = Lux.apply(
        enc.logvar_head,
        h,
        ps.logvar,
        st_new.logvar
    )
    @reset st_new.logvar = st_logvar

    μ = reshape(μ_flat, Q, P, S)
    logvar = reshape(logvar_flat, Q, P, S)
    logvar = clamp.(logvar, -10.0f0, 2.0f0)

    if !isnothing(component_mask)
        μ_selected = dropdims(sum(component_mask .* μ; dims = 2); dims = 2)
        logvar_selected = dropdims(sum(component_mask .* logvar; dims = 2); dims = 2)
        ε_reshaped = reshape(ε[1:Q, :], Q, S)

        σ = exp.(0.5f0 .* logvar_selected)
        z = μ_selected .+ σ .* ε_reshaped

        log_q = -0.5f0 .* sum(
            logvar_selected .+ (z .- μ_selected) .^ 2 ./
                exp.(logvar_selected) .+ log(2.0f0 * Float32(π));
            dims = 1
        )
        log_q = dropdims(log_q; dims = 1)

        z = reshape(z, Q, 1, S)
    else
        ε_reshaped = reshape(ε, Q, P, S)
        σ = exp.(0.5f0 .* logvar)
        z = μ .+ σ .* ε_reshaped

        log_q = -0.5f0 .* sum(
            logvar .+ (z .- μ) .^ 2 ./ exp.(logvar) .+ log(2.0f0 * Float32(π));
            dims = (1, 2)
        )
        log_q = dropdims(log_q; dims = (1, 2))
    end

    return z, log_q, μ, logvar, st_new
end

end
