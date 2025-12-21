module RefPriors

export prior_map,
    UniformPrior, GaussianPrior, LogNormalPrior, LearnableGaussianPrior, EbmPrior

using ..Utils

struct UniformPrior{T <: Float32} <: AbstractPrior
    ε::T
end
struct GaussianPrior{T <: Float32} <: AbstractPrior
    ε::T
end
struct LogNormalPrior{T <: Float32} <: AbstractPrior
    ε::T
end
struct LearnableGaussianPrior{T <: Float32} <: AbstractPrior
    ε::T
end
struct EbmPrior{T <: Float32} <: AbstractPrior
    ε::T
end

function stable_log(pdf, ε)
    return log.(pdf .+ ε)
end

function (prior::UniformPrior)(
        z,
        π_μ,
        π_σ;
        log_bool = false,
    )
    z = @. (z >= 0) * (z <= 1)
    z = log_bool ? stable_log(z, prior.ε) : z
    return z
end

function (prior::GaussianPrior)(
        z,
        π_μ,
        π_σ;
        log_bool = false,
    )
    scale = Float32(1 / sqrt(2π))
    z = scale .* exp.(-z .^ 2 ./ 2)
    z = log_bool ? stable_log(z, prior.ε) : z
    return z
end

function (prior::LogNormalPrior)(
        z,
        π_μ,
        π_σ;
        log_bool = false,
    )
    sqrt_2π = Float32(sqrt(2π))
    z = exp.(-((log.(z .+ prior.ε))) ./ 2) ./ (z .* sqrt_2π .* prior.ε)
    z = log_bool ? stable_log(z, prior.ε) : z
    return z
end

function (prior::LearnableGaussianPrior)(
        z,
        π_μ,
        π_σ;
        log_bool = false,
    )
    π_eps = π_σ .* Float32(sqrt(2π)) .+ prior.ε
    denom_eps = 2 .* π_σ .^ 2 .+ prior.ε
    z = (1 / abs(π_eps)) .* exp.(-((z .- π_μ) .^ 2) ./ denom_eps)
    z = log_bool ? stable_log(z, prior.ε) : z
    return z
end

function (prior::EbmPrior)(
        z,
        π_μ,
        π_σ;
        log_bool = false,
    )
    log_pdf = zero(z)
    log_pdf = log_bool ? log_pdf .- 1.0f0 : log_pdf
    return log_pdf .+ 1.0f0
end

const prior_map = Dict(
    "uniform" => ε -> UniformPrior(ε),
    "gaussian" => ε -> GaussianPrior(ε),
    "lognormal" => ε -> LogNormalPrior(ε),
    "learnable_gaussian" => ε -> LearnableGaussianPrior(ε),
    "ebm" => ε -> EbmPrior(ε),
)

end
