module Wavelets

export DoGWavelet,
    MHWavelet,
    MorletWavelet,
    ShannonWavelet

using ..Utils: AbstractBasis

# Derivative of gaussian
struct DoGWavelet <: AbstractBasis
end

function (w::DoGWavelet)(x, τ)
    exp_term = exp.(- x ./ 2)
    y = x .* exp_term
    y = y ./ sqrt(2π)
    return y
end

# Mexican Hat
struct MHWavelet <: AbstractBasis
end

function (w::MHWavelet)(x, τ)
    term_1 = x .^ 2 .- 1
    term_2 = exp.(- x .^ 2 ./ 2)
    y = term_1 .* term_2
    y = y .* 2 / sqrt(3 * sqrt(π))
    return y
end

# Morlet
struct MorletWavelet <: AbstractBasis
end

function (w::MorletWavelet)(x, τ)
    real = cos.(τ .* x)
    envelope = exp.(-x .^ 2 ./ 2)
    y = real .* envelope
    return y
end

# ShannonWavelet
struct ShannonWavelet
end

function (w::ShannonWavelet)(x, τ)
    first_term = sinc.(x .* 2.0f0 .* π)
    second_term = cos.(x .* π / 3.0f0)
    y = first_term .* second_term
    return y .* 2.0f0
end

end
