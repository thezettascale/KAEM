module Losses

export IS_loss, MALA_loss

using ..Utils

using CUDA, KernelAbstractions, Tullio
using NNlib: conv, meanpool

perceptual_loss = parse(Bool, get(ENV, "PERCEPTUAL", "true"))
feature_extractor = nothing
style_lyrs = [2, 5, 9, 12]
content_lyrs = [9]
if perceptual_loss
    using Metalhead: VGG
    feature_extractor = VGG(16; pretrain = true).layers[1][1:12] |> pu # Conv layers only, (rest is classifier)
end


## Fcns for model with Importance Sampling ##
function cross_entropy_IS(
    x::AbstractArray{T,3},
    x̂::AbstractArray{T,4},
    ε::T,
    scale::T,
)::AbstractArray{T,2} where {T<:half_quant}
    x̂ = x̂ .+ ε
    @tullio ll[b, s] := log(x̂[d, t, s, b]) * x[d, t, b]
    return ll ./ size(x̂, 1) ./ scale
end

function l2_IS(
    x::AbstractArray{T,4},
    x̂::AbstractArray{T,5},
    ε::T,
    scale::T,
)::AbstractArray{T,2} where {T<:half_quant}
    @tullio ll[b, s] := - (x[w, h, c, b] - x̂[w, h, c, s, b]) ^ 2
    return ll ./ scale
end

function l2_IS_PCA(
    x::AbstractArray{T,2},
    x̂::AbstractArray{T,3},
    ε::T,
    scale::T,
)::AbstractArray{T,2} where {T<:half_quant}
    @tullio ll[b, s] := - (x[d, b] - x̂[d, s, b]) ^ 2
    return ll ./ scale
end

function IS_loss(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
    B::Int,
    S::Int,
    SEQ::Bool,
)::AbstractArray{T,2} where {T<:half_quant}
    loss_fcn = (SEQ ? cross_entropy_IS : (ndims(x) == 2 ? l2_IS_PCA : l2_IS))
    return loss_fcn(x, x̂, ε, scale)
end

## Fcns for model with Langevin methods ##
function cross_entropy_MALA(
    x::AbstractArray{T,3},
    x̂::AbstractArray{T,3},
    ε::T,
    scale::T,
)::AbstractArray{T,1} where {T<:half_quant}
    x̂ = x̂ .+ ε
    @tullio ll[b] := log(x̂[d, t, b]) * x[d, t, b]
    return ll ./ size(x, 1) ./ scale
end

function l2_PCA(
    x::AbstractArray{T,2},
    x̂::AbstractArray{T,2},
    ε::T,
    scale::T,
    perceptual_scale::T,
)::AbstractArray{T,1} where {T<:half_quant}
    @tullio ll[b] := - (x[d, b] - x̂[d, b]) ^ 2
    return ll ./ scale
end

function l2_MALA(
    x::AbstractArray{T,4},
    x̂::AbstractArray{T,4},
    ε::T,
    scale::T,
    perceptual_scale::T,
)::AbstractArray{T,1} where {T<:half_quant}
    @tullio ll[b] := - (x[w, h, c, b] - x̂[w, h, c, b]) ^ 2
    return ll ./ scale
end

## SSIM ##
window_size = 11
σ = half_quant(1.5)
channels = 3
C1 = half_quant(0.01^2)
C2 = half_quant(0.03^2)

radius = (window_size - 1) ÷ 2
ax = collect((-radius):radius)
g = exp.(-(ax .^ 2) ./ (2 * σ^2))
g ./= sum(g)
window = g * g' .|> half_quant
window ./= sum(window)
pad = (window_size - 1) ÷ 2

kernel = reshape(window, window_size, window_size, 1, 1)
kernel = repeat(kernel, 1, 1, channels, 1) |> pu

function ssim(
    x::AbstractArray{T,4},
    x̂::AbstractArray{T,4},
)::AbstractArray{T,1} where {T<:half_quant}
    H, W, C, B = size(x)
    μ_x = conv(x, kernel; pad = pad)
    μ_y = conv(x̂, kernel; pad = pad)
    μ_x2, μ_y2, μ_xy = μ_x .^ 2, μ_y .^ 2, μ_x .* μ_y

    σ_x2 = conv(x .^ 2, kernel; pad = pad) .- μ_x2
    σ_y2 = conv(x̂ .^ 2, kernel; pad = pad) .- μ_y2
    σ_xy = conv(x .* x̂, kernel; pad = pad) .- μ_xy

    num = (2 .* μ_xy .+ C1) .* (2 .* σ_xy .+ C2)
    den = (μ_x2 .+ μ_y2 .+ C1) .* (σ_x2 .+ σ_y2 .+ C2)
    @tullio ll[b] := num[w, h, c, b] / den[w, h, c, b]
    return ll ./ (H * W * C)
end


function ssim_MALA(
    x::AbstractArray{T,4},
    x̂::AbstractArray{T,4},
    ε::T,
    scale::T,
    perceptual_scale::T,
)::AbstractArray{T,1} where {T<:half_quant}
    return exp.(ssim(x, x̂) ./ scale)
end

function gramm_loss(
    x::AbstractArray{T,4},
    x̂::AbstractArray{T,4},
    scale::T,
)::AbstractArray{T,1} where {T<:half_quant}
    H, W, C, B = size(x)
    real = reshape(x, H * W, C, B)
    fake = reshape(x̂, H * W, C, B)
    @tullio G_real[c1, c2, b] := real[p, c1, b] * real[p, c2, b]
    @tullio G_fake[c1, c2, b] := fake[p, c1, b] * fake[p, c2, b]
    @tullio ll[b] := - (G_real[c1, c2, b] - G_fake[c1, c2, b]) ^ 2
    return ll ./ scale
end

function feature_loss(
    x::AbstractArray{T,4},
    x̂::AbstractArray{T,4},
    ε::T,
    scale::T,
    perceptual_scale::T,
)::AbstractArray{T,1} where {T<:half_quant}
    loss = l2_MALA(x, x̂, ε, scale, perceptual_scale)
    real_features, fake_features = x, x̂
    for (idx, layer) in enumerate(feature_extractor)
        scale_at_lyr = prod(size(real_features)[1:3]) * scale
        real_features, fake_features = layer(real_features), layer(fake_features)
        loss =
            (idx in style_lyrs) ?
            perceptual_scale .* gramm_loss(real_features, fake_features, scale_at_lyr) +
            loss : loss

        loss =
            (idx in content_lyrs) ?
            perceptual_scale .*
            l2_MALA(real_features, fake_features, ε, scale_at_lyr, perceptual_scale) +
            loss : loss
    end
    return loss
end

function MALA_loss(
    x::AbstractArray{T},
    x̂::AbstractArray{T},
    ε::T,
    scale::T,
    B::Int,
    SEQ::Bool,
    perceptual_scale::T,
)::AbstractArray{T,1} where {T<:half_quant}
    loss_fcn = (
        SEQ ? cross_entropy_MALA :
        (ndims(x) == 2 ? l2_PCA : (perceptual_loss ? feature_loss : ssim_MALA))
    )
    return loss_fcn(x, x̂, ε, scale, perceptual_scale)
end

end
