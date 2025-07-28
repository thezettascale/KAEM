module RefPriors

export prior_map,
    UniformPrior, GaussianPrior, LogNormalPrior, LearnableGaussianPrior, EbmPrior

using CUDA, ParallelStencil, Lux

using ..Utils

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, half_quant, 3)
else
    @init_parallel_stencil(Threads, half_quant, 3)
end

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

@parallel_indices (q, p, b) function uniform_pdf!(
    pdf::AbstractArray{T},
    z::AbstractArray{T},
)::Nothing where {T<:half_quant}
    pdf[q, p, b] = T((z[q, p, b] >= zero(T)) * (z[q, p, b] <= one(T)))
    return nothing
end

@parallel_indices (q, p, b) function gaussian_pdf!(
    pdf::AbstractArray{T},
    z::AbstractArray{T},
)::Nothing where {T<:half_quant}
    pdf[q, p, b] = T(1 / sqrt(2π)) * exp(-z[q, p, b]^2 / 2)
    return nothing
end

@parallel_indices (q, p, b) function lognormal_pdf!(
    pdf::AbstractArray{T},
    z::AbstractArray{T},
    ε::T,
)::Nothing where {T<:half_quant}
    pdf[q, p, b] = exp(-(log(z[q, p, b] + ε))^2 / 2) / (z[q, p, b] * T(sqrt(2π)) + ε)
    return nothing
end

@parallel_indices (q, p, b) function learnable_gaussian_pdf!(
    pdf::AbstractArray{T},
    z::AbstractArray{T},
    ε::T,
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T},
)::Nothing where {T<:half_quant}
    pdf[q, p, b] =
        one(T) / (
            abs(π_σ[p] * T(sqrt(2π)) + ε) *
            exp(-(z[q, p, b] - π_μ[p]^2) / (2 * (π_σ[p]^2) + ε))
        )
    return nothing
end

@parallel_indices (q, p, b) function ebm_pdf!(
    pdf::AbstractArray{T},
    z::AbstractArray{T},
    ε::T,
)::Nothing where {T<:half_quant}
    pdf[q, p, b] = one(T) - ε # Minus eps to counter + eps in stable log
    return nothing
end

function (prior::UniformPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T};
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    Q, P, S = size(z)
    log_pdf = @zeros(Q, P, S)
    @parallel (1:Q, 1:P, 1:S) uniform_pdf!(log_pdf, z)
    log_bool && @parallel (1:Q, 1:P, 1:S) stable_log!(log_pdf, prior.ε)
    return log_pdf
end

function (prior::GaussianPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T};
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    Q, P, S = size(z)
    log_pdf = @zeros(Q, P, S)
    @parallel (1:Q, 1:P, 1:S) gaussian_pdf!(log_pdf, z)
    log_bool && @parallel (1:Q, 1:P, 1:S) stable_log!(log_pdf, prior.ε)
    return log_pdf
end

function (prior::LogNormalPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T},
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    Q, P, S = size(z)
    log_pdf = @zeros(Q, P, S)
    @parallel (1:Q, 1:P, 1:S) lognormal_pdf!(log_pdf, z, prior.ε)
    log_bool && @parallel (1:Q, 1:P, 1:S) stable_log!(log_pdf, prior.ε)
    return log_pdf
end

function (prior::LearnableGaussianPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T},
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    Q, P, S = size(z)
    log_pdf = @zeros(Q, P, S)
    @parallel (1:Q, 1:P, 1:S) learnable_gaussian_pdf!(log_pdf, z, prior.ε, π_μ, π_σ)
    log_bool && @parallel (1:Q, 1:P, 1:S) stable_log!(log_pdf, prior.ε)
    return log_pdf
end

function (prior::EbmPrior)(
    z::AbstractArray{T},
    π_μ::AbstractArray{T},
    π_σ::AbstractArray{T},
    log_bool::Bool = false,
)::AbstractArray{T} where {T<:half_quant}
    Q, P, S = size(z)
    log_pdf = @zeros(Q, P, S)
    @parallel (1:Q, 1:P, 1:S) ebm_pdf!(log_pdf, z, prior.ε)
    log_bool && @parallel (1:Q, 1:P, 1:S) stable_log!(log_pdf, prior.ε)
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
