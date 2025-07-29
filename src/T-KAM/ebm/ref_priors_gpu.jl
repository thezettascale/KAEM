module RefPriors

export prior_map,
    UniformPrior, GaussianPrior, LogNormalPrior, LearnableGaussianPrior, EbmPrior

using CUDA, Lux, KernelAbstractions, Tullio

using ..Utils

struct UniformPrior <: Lux.AbstractLuxLayer
    ε::half_quant
end
struct GaussianPrior <: Lux.AbstractLuxLayer
    ε::half_quant
end
struct LogNormalPrior <: Lux.AbstractLuxLayer
    ε::half_quant
end
struct LearnableGaussianPrior <: Lux.AbstractLuxLayer
    ε::half_quant
end
struct EbmPrior <: Lux.AbstractLuxLayer
    ε::half_quant
end

function stable_log(pdf::AbstractArray{T}, ε::T)::AbstractArray{T} where {T<:half_quant}
    return log.(pdf .+ ε)
end

function (prior::UniformPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T};
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    @tullio pdf[q, p, s] := (z[q, p, s] >= 0) * (z[q, p, s] <= 1)
    pdf = T.(pdf)
    log_bool && return stable_log(pdf, prior.ε)
    return pdf
end

function (prior::GaussianPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T};
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    scale = T(1 / sqrt(2π))
    @tullio pdf[q, p, s] := scale * exp(-z[q, p, s]^2 / 2)
    pdf = T.(pdf)
    log_bool && return stable_log(pdf, prior.ε)
    return pdf
end

function (prior::LogNormalPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T};
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    sqrt_2π = T(sqrt(2π))
    @tullio pdf[q, p, s] := exp(-(log(z[q, p, s] + prior.ε))^2 / 2) / (z[q, p, s] * sqrt_2π + prior.ε)
    pdf = T.(pdf)
    log_bool && return stable_log(pdf, prior.ε)
    return pdf
end

function (prior::LearnableGaussianPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T};
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    sqrt_2π = T(sqrt(2π))
    @tullio pdf[q, p, s] := 1 / (abs(π_σ[q, p] * sqrt_2π + prior.ε) * exp(-(z[q, p, s] - π_μ[q, p]^2) / (2 * (π_σ[q, p]^2) + prior.ε)))
    pdf = T.(pdf)
    log_bool && return stable_log(pdf, prior.ε)
    return pdf
end

function (prior::EbmPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T};
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    @tullio log_pdf[q, p, s] := 0 * z[q, p, s] + 1
    log_bool && return stable_log(log_pdf, prior.ε)
    return log_pdf
end

const prior_map = Dict(
    "uniform" => ε -> UniformPrior(ε),
    "gaussian" => ε -> GaussianPrior(ε),
    "lognormal" => ε -> LogNormalPrior(ε),
    "learnable_gaussian" => ε -> LearnableGaussianPrior(ε),
    "ebm" => ε -> EbmPrior(ε),
)

end
