module Generator

export PangGenerator, init_pang_generator

using Lux, Accessors, NNlib, Random

using ..Utils

struct PangGenerator <: Lux.AbstractLuxLayer
    depth::Int
    project::Lux.Dense
    conv_layers::Tuple{Vararg{Lux.ConvTranspose}}
    init_spatial::Int
    init_channels::Int
end

function init_pang_generator(
        x_shape::Tuple{Vararg{Int}},
        latent_dim::Int,
        gen_channels::Vector{Int},
        strides::Vector{Int},
        kernels::Vector{Int},
        paddings::Vector{Int},
    )
    in_channels = last(x_shape)
    img_size = first(x_shape)
    num_upsample = count(s -> s == 2, strides)
    init_spatial = img_size ÷ (2^num_upsample)
    init_channels = first(gen_channels)

    project = Lux.Dense(
        latent_dim,
        init_channels * init_spatial * init_spatial,
        NNlib.relu
    )

    gen_conv_layers = Vector{Lux.ConvTranspose}()
    prev_c = init_channels

    for (i, c) in enumerate(gen_channels)
        is_last = i == length(gen_channels)
        out_c = is_last ? in_channels : c
        push!(
            gen_conv_layers,
            Lux.ConvTranspose(
                (kernels[i], kernels[i]),
                prev_c => out_c;
                stride = strides[i],
                pad = paddings[i],
            ),
        )
        prev_c = c
    end

    return PangGenerator(
        length(gen_conv_layers),
        project,
        Tuple(gen_conv_layers),
        init_spatial,
        init_channels,
    )
end

function Lux.initialparameters(rng::AbstractRNG, gen::PangGenerator)
    gen_conv_ps = NamedTuple(
        symbol_map[i] => Lux.initialparameters(rng, gen.conv_layers[i])
            for i in 1:gen.depth
    )

    return (
        project = Lux.initialparameters(rng, gen.project),
        conv = gen_conv_ps,
    )
end

function Lux.initialstates(rng::AbstractRNG, gen::PangGenerator)
    gen_conv_st = NamedTuple(
        symbol_map[i] => Lux.initialstates(rng, gen.conv_layers[i]) |> Lux.f32
            for i in 1:gen.depth
    )

    return (
        project = Lux.initialstates(rng, gen.project) |> Lux.f32,
        conv = gen_conv_st,
    )
end

function (gen::PangGenerator)(z, ps, st)
    h, st_proj = Lux.apply(gen.project, z, ps.project, st.project)
    st_new = st
    @reset st_new.project = st_proj

    h = reshape(h, gen.init_spatial, gen.init_spatial, gen.init_channels, size(z, 2))

    for i in 1:gen.depth
        h, st_layer = Lux.apply(
            gen.conv_layers[i], h, ps.conv[symbol_map[i]], st.conv[symbol_map[i]]
        )
        @reset st_new.conv[symbol_map[i]] = st_layer

        is_last = i == gen.depth
        if is_last
            h = NNlib.sigmoid(h)
        else
            h = NNlib.leakyrelu(h)
        end
    end

    return h, st_new
end

end
