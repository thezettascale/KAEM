module DDPMArchitecture

export TimeEmbedding, DownBlock, UpBlock, UNet, DDPMConfig
export init_time_embedding, init_down_block, init_up_block
export apply_down_block, apply_up_block, sinusoidal_embedding

using Lux, NNlib, Accessors, Random

using ..Utils

struct DDPMConfig <: AbstractBoolConfig
    batchnorm::Bool
    attention::Bool
end

#= Time Embedding =#

struct TimeEmbedding <: Lux.AbstractLuxLayer
    dim::Int
    hidden_dim::Int
    dense1::Lux.Dense
    dense2::Lux.Dense
end

function init_time_embedding(dim::Int, hidden_dim::Int)
    dense1 = Lux.Dense(dim, hidden_dim, NNlib.gelu)
    dense2 = Lux.Dense(hidden_dim, hidden_dim)
    return TimeEmbedding(dim, hidden_dim, dense1, dense2)
end

function Lux.initialparameters(rng::AbstractRNG, te::TimeEmbedding)
    return (
        dense1 = Lux.initialparameters(rng, te.dense1),
        dense2 = Lux.initialparameters(rng, te.dense2),
    )
end

function Lux.initialstates(rng::AbstractRNG, te::TimeEmbedding)
    return (
        dense1 = Lux.initialstates(rng, te.dense1) |> Lux.f32,
        dense2 = Lux.initialstates(rng, te.dense2) |> Lux.f32,
    )
end

function sinusoidal_embedding(t::AbstractArray{T}, dim::Int) where {T}
    half_dim = dim ÷ 2
    emb_scale = log(10000.0f0) / (half_dim - 1)
    emb = exp.(-(0:(half_dim - 1)) .* emb_scale) |> x -> reshape(x, 1, :)
    emb = t .* emb
    return cat(sin.(emb), cos.(emb), dims = 2)
end

function (te::TimeEmbedding)(t, ps, st)
    emb = sinusoidal_embedding(t, te.dim)
    h, st1 = Lux.apply(te.dense1, emb', ps.dense1, st.dense1)
    h, st2 = Lux.apply(te.dense2, h, ps.dense2, st.dense2)
    return h, (dense1 = st1, dense2 = st2)
end

#= Down Block =#

struct DownBlock <: Lux.AbstractLuxLayer
    conv1::Lux.Conv
    conv2::Lux.Conv
    time_proj::Lux.Dense
    norm1::Lux.GroupNorm
    norm2::Lux.GroupNorm
    downsample::Lux.Conv
    out_channels::Int
end

function init_down_block(in_c::Int, out_c::Int, time_dim::Int, kernel_size::Int)
    return DownBlock(
        Lux.Conv((kernel_size, kernel_size), in_c => out_c; pad = 1),
        Lux.Conv((kernel_size, kernel_size), out_c => out_c; pad = 1),
        Lux.Dense(time_dim, out_c),
        Lux.GroupNorm(out_c, min(8, out_c)),
        Lux.GroupNorm(out_c, min(8, out_c)),
        Lux.Conv((kernel_size, kernel_size), out_c => out_c; stride = 2, pad = 1),
        out_c,
    )
end

function apply_down_block(block::DownBlock, h, t_emb, ps, st)
    st_new = st

    h, st_c1 = Lux.apply(block.conv1, h, ps.conv1, st.conv1)
    @reset st_new.conv1 = st_c1
    h = NNlib.gelu(h)

    t_proj, st_tp = Lux.apply(block.time_proj, t_emb, ps.time_proj, st.time_proj)
    @reset st_new.time_proj = st_tp
    h = h .+ reshape(t_proj, 1, 1, :, size(t_proj, 2))

    h, st_n1 = Lux.apply(block.norm1, h, ps.norm1, st.norm1)
    @reset st_new.norm1 = st_n1

    h, st_c2 = Lux.apply(block.conv2, h, ps.conv2, st.conv2)
    @reset st_new.conv2 = st_c2

    h, st_n2 = Lux.apply(block.norm2, h, ps.norm2, st.norm2)
    @reset st_new.norm2 = st_n2
    h = NNlib.gelu(h)

    h_down, st_ds = Lux.apply(block.downsample, h, ps.downsample, st.downsample)
    @reset st_new.downsample = st_ds

    return h_down, h, st_new  # Return both downsampled and skip connection
end

#= Up Block =#

struct UpBlock <: Lux.AbstractLuxLayer
    conv1::Lux.Conv
    conv2::Lux.Conv
    time_proj::Lux.Dense
    norm1::Lux.GroupNorm
    norm2::Lux.GroupNorm
    upsample::Lux.ConvTranspose
    out_channels::Int
end

function init_up_block(in_c::Int, out_c::Int, skip_c::Int, time_dim::Int, kernel_size::Int)
    return UpBlock(
        Lux.Conv(
            (kernel_size, kernel_size),
            in_c + skip_c => out_c;
            pad = 1
        ),
        Lux.Conv(
            (kernel_size, kernel_size),
            out_c => out_c;
            pad = 1
        ),
        Lux.Dense(time_dim, out_c),
        Lux.GroupNorm(out_c, min(8, out_c)),
        Lux.GroupNorm(out_c, min(8, out_c)),
        Lux.ConvTranspose(
            (kernel_size, kernel_size),
            in_c => in_c; stride = 2,
            pad = 1,
            outpad = 1
        ),
        out_c,
    )
end

function apply_up_block(block::UpBlock, h, skip, t_emb, ps, st)
    st_new = st

    h, st_us = Lux.apply(block.upsample, h, ps.upsample, st.upsample)
    @reset st_new.upsample = st_us

    h = cat(h, skip, dims = 3)

    h, st_c1 = Lux.apply(block.conv1, h, ps.conv1, st.conv1)
    @reset st_new.conv1 = st_c1
    h = NNlib.gelu(h)

    t_proj, st_tp = Lux.apply(block.time_proj, t_emb, ps.time_proj, st.time_proj)
    @reset st_new.time_proj = st_tp
    h = h .+ reshape(t_proj, 1, 1, :, size(t_proj, 2))

    h, st_n1 = Lux.apply(block.norm1, h, ps.norm1, st.norm1)
    @reset st_new.norm1 = st_n1

    h, st_c2 = Lux.apply(block.conv2, h, ps.conv2, st.conv2)
    @reset st_new.conv2 = st_c2

    h, st_n2 = Lux.apply(block.norm2, h, ps.norm2, st.norm2)
    @reset st_new.norm2 = st_n2
    h = NNlib.gelu(h)

    return h, st_new
end

#= UNet =#

struct UNet <: Lux.AbstractLuxLayer
    time_embed::TimeEmbedding
    init_conv::Lux.Conv
    down_blocks::Tuple{Vararg{DownBlock}}
    mid_conv1::Lux.Conv
    mid_conv2::Lux.Conv
    mid_norm::Lux.GroupNorm
    up_blocks::Tuple{Vararg{UpBlock}}
    final_conv::Lux.Conv
    num_down::Int
    num_up::Int
end

function init_unet(
        in_channels::Int,
        channels::Vector{Int},
        kernel_size::Int,
        time_dim::Int,
    )
    time_embed = init_time_embedding(time_dim, time_dim * 4)
    init_conv = Lux.Conv((3, 3), in_channels => first(channels); pad = 1)

    # Down blocks
    down_blocks = Vector{DownBlock}()
    prev_c = first(channels)
    for c in channels[2:end]
        push!(down_blocks, init_down_block(prev_c, c, time_dim * 4, kernel_size))
        prev_c = c
    end

    # Middle
    mid_c = last(channels)
    mid_conv1 = Lux.Conv(
        (kernel_size, kernel_size),
        mid_c => mid_c;
        pad = 1
    )
    mid_conv2 = Lux.Conv(
        (kernel_size, kernel_size),
        mid_c => mid_c;
        pad = 1
    )
    mid_norm = Lux.GroupNorm(mid_c, min(8, mid_c))

    # Up blocks
    up_blocks = Vector{UpBlock}()
    rev_channels = reverse(channels)
    for i in 1:(length(rev_channels) - 1)
        in_c = rev_channels[i]
        out_c = rev_channels[i + 1]
        skip_c = in_c
        push!(
            up_blocks,
            init_up_block(
                in_c, out_c, skip_c, time_dim * 4,
                kernel_size
            )
        )
    end

    final_conv = Lux.Conv((1, 1), first(channels) => in_channels)

    return UNet(
        time_embed,
        init_conv,
        Tuple(down_blocks),
        mid_conv1,
        mid_conv2,
        mid_norm,
        Tuple(up_blocks),
        final_conv,
        length(down_blocks),
        length(up_blocks),
    )
end

function Lux.initialparameters(rng::AbstractRNG, unet::UNet)
    down_ps = NamedTuple(
        symbol_map[i] => (
                conv1 = Lux.initialparameters(rng, unet.down_blocks[i].conv1),
                conv2 = Lux.initialparameters(rng, unet.down_blocks[i].conv2),
                time_proj = Lux.initialparameters(rng, unet.down_blocks[i].time_proj),
                norm1 = Lux.initialparameters(rng, unet.down_blocks[i].norm1),
                norm2 = Lux.initialparameters(rng, unet.down_blocks[i].norm2),
                downsample = Lux.initialparameters(rng, unet.down_blocks[i].downsample),
            ) for i in 1:unet.num_down
    )

    up_ps = NamedTuple(
        symbol_map[i] => (
                conv1 = Lux.initialparameters(rng, unet.up_blocks[i].conv1),
                conv2 = Lux.initialparameters(rng, unet.up_blocks[i].conv2),
                time_proj = Lux.initialparameters(rng, unet.up_blocks[i].time_proj),
                norm1 = Lux.initialparameters(rng, unet.up_blocks[i].norm1),
                norm2 = Lux.initialparameters(rng, unet.up_blocks[i].norm2),
                upsample = Lux.initialparameters(rng, unet.up_blocks[i].upsample),
            ) for i in 1:unet.num_up
    )

    return (
        time_embed = Lux.initialparameters(rng, unet.time_embed),
        init_conv = Lux.initialparameters(rng, unet.init_conv),
        down = down_ps,
        mid_conv1 = Lux.initialparameters(rng, unet.mid_conv1),
        mid_conv2 = Lux.initialparameters(rng, unet.mid_conv2),
        mid_norm = Lux.initialparameters(rng, unet.mid_norm),
        up = up_ps,
        final_conv = Lux.initialparameters(rng, unet.final_conv),
    )
end

function Lux.initialstates(rng::AbstractRNG, unet::UNet)
    down_st = NamedTuple(
        symbol_map[i] => (
                conv1 = Lux.initialstates(rng, unet.down_blocks[i].conv1) |> Lux.f32,
                conv2 = Lux.initialstates(rng, unet.down_blocks[i].conv2) |> Lux.f32,
                time_proj = Lux.initialstates(rng, unet.down_blocks[i].time_proj) |> Lux.f32,
                norm1 = Lux.initialstates(rng, unet.down_blocks[i].norm1) |> Lux.f32,
                norm2 = Lux.initialstates(rng, unet.down_blocks[i].norm2) |> Lux.f32,
                downsample = Lux.initialstates(rng, unet.down_blocks[i].downsample) |> Lux.f32,
            ) for i in 1:unet.num_down
    )

    up_st = NamedTuple(
        symbol_map[i] => (
                conv1 = Lux.initialstates(rng, unet.up_blocks[i].conv1) |> Lux.f32,
                conv2 = Lux.initialstates(rng, unet.up_blocks[i].conv2) |> Lux.f32,
                time_proj = Lux.initialstates(rng, unet.up_blocks[i].time_proj) |> Lux.f32,
                norm1 = Lux.initialstates(rng, unet.up_blocks[i].norm1) |> Lux.f32,
                norm2 = Lux.initialstates(rng, unet.up_blocks[i].norm2) |> Lux.f32,
                upsample = Lux.initialstates(rng, unet.up_blocks[i].upsample) |> Lux.f32,
            ) for i in 1:unet.num_up
    )

    return (
        time_embed = Lux.initialstates(rng, unet.time_embed),
        init_conv = Lux.initialstates(rng, unet.init_conv) |> Lux.f32,
        down = down_st,
        mid_conv1 = Lux.initialstates(rng, unet.mid_conv1) |> Lux.f32,
        mid_conv2 = Lux.initialstates(rng, unet.mid_conv2) |> Lux.f32,
        mid_norm = Lux.initialstates(rng, unet.mid_norm) |> Lux.f32,
        up = up_st,
        final_conv = Lux.initialstates(rng, unet.final_conv) |> Lux.f32,
    )
end

function (unet::UNet)(x_noisy, t, ps, st)
    st_new = st

    # Time embedding
    t_emb, st_te = unet.time_embed(t, ps.time_embed, st.time_embed)
    @reset st_new.time_embed = st_te

    # Initial conv
    h, st_ic = Lux.apply(unet.init_conv, x_noisy, ps.init_conv, st.init_conv)
    @reset st_new.init_conv = st_ic
    h = NNlib.gelu(h)

    # Down path with skip connections
    skips = Vector{typeof(h)}()
    for i in 1:unet.num_down
        h, skip, st_down = apply_down_block(
            unet.down_blocks[i], h, t_emb, ps.down[symbol_map[i]], st.down[symbol_map[i]]
        )
        @reset st_new.down[symbol_map[i]] = st_down
        push!(skips, skip)
    end

    # Middle
    h, st_mc1 = Lux.apply(unet.mid_conv1, h, ps.mid_conv1, st.mid_conv1)
    @reset st_new.mid_conv1 = st_mc1
    h = NNlib.gelu(h)

    h, st_mn = Lux.apply(unet.mid_norm, h, ps.mid_norm, st.mid_norm)
    @reset st_new.mid_norm = st_mn

    h, st_mc2 = Lux.apply(unet.mid_conv2, h, ps.mid_conv2, st.mid_conv2)
    @reset st_new.mid_conv2 = st_mc2
    h = NNlib.gelu(h)

    # Up path
    for i in 1:unet.num_up
        skip = skips[end - i + 1]
        h, st_up = apply_up_block(
            unet.up_blocks[i], h, skip, t_emb, ps.up[symbol_map[i]], st.up[symbol_map[i]]
        )
        @reset st_new.up[symbol_map[i]] = st_up
    end

    # Final conv
    noise_pred, st_fc = Lux.apply(unet.final_conv, h, ps.final_conv, st.final_conv)
    @reset st_new.final_conv = st_fc

    return noise_pred, st_new
end

end
