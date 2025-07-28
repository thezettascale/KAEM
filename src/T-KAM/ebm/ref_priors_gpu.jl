module RefPriors

export prior_map,
    UniformPrior, GaussianPrior, LogNormalPrior, LearnableGaussianPrior, EbmPrior

using CUDA, Lux

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

function (prior::UniformPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T};
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    pdf = T.((z .>= zero(T)) .* (z .<= one(T)))
    log_bool && return log.(pdf .+ prior.ε)
    return pdf
end

function (prior::GaussianPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T};
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    pdf = T(1 ./ sqrt(2π)) .* exp.(-z .^ 2 ./ 2)
    log_bool && return log.(pdf .+ prior.ε)
    return pdf
end

function (prior::LogNormalPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T};
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    pdf = exp.(-(log.(z .+ prior.ε)) .^ 2 ./ 2) ./ (z .* T(sqrt(2π)) .+ prior.ε)
    log_bool && return log.(pdf .+ prior.ε)
    return pdf
end

function (prior::LearnableGaussianPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T};
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    pdf = one(T) ./ (
        abs.(π_σ .* T(sqrt(2π)) .+ prior.ε) .*
        exp.(-(z .- π_μ .^ 2) ./ (2 .* (π_σ .^ 2) .+ prior.ε))
    )
    log_bool && return log.(pdf .+ prior.ε)
    return pdf
end

function (prior::EbmPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T};
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    log_pdf = zero(T) .* z
    log_bool && return log_pdf
    return log_pdf .+ one(T)
end

const prior_map = Dict(
    "uniform" => ε -> UniformPrior(ε),
    "gaussian" => ε -> GaussianPrior(ε),
    "lognormal" => ε -> LogNormalPrior(ε),
    "learnable_gaussian" => ε -> LearnableGaussianPrior(ε),
    "ebm" => ε -> EbmPrior(ε),
)

end
