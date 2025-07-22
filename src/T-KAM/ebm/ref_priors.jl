module RefPriors

export prior_pdf

using CUDA, KernelAbstractions, ParallelStencil

include("../../utils.jl")
using .Utils: half_quant

@static if CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))
    @init_parallel_stencil(CUDA, half_quant, 3)
else
    @init_parallel_stencil(Threads, half_quant, 3)
end

@parallel_indices (q, p, b) function uniform_pdf!(
    pdf::AbstractArray{T},
    z::AbstractArray{T},
    ε::T,
    _π_μ::AbstractArray{T},
    _π_σ::AbstractArray{T},
)::Nothing where {T<:half_quant}
    pdf[q, p, b] = T((z[q, p, b] >= zero(T)) * (z[q, p, b] <= one(T)))
    return nothing
end

@parallel_indices (q, p, b) function gaussian_pdf!(
    pdf::AbstractArray{T},
    z::AbstractArray{T},
    ε::T,
    _π_μ::AbstractArray{T},
    _π_σ::AbstractArray{T},
)::Nothing where {T<:half_quant}
    pdf[q, p, b] = T(1 / sqrt(2π)) * exp(-z[q, p, b]^2 / 2)
    return nothing
end

@parallel_indices (q, p, b) function lognormal_pdf!(
    pdf::AbstractArray{T},
    z::AbstractArray{T},
    ε::T,
    _π_μ::AbstractArray{T},
    _π_σ::AbstractArray{T},
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
    _π_μ::AbstractArray{T},
    _π_σ::AbstractArray{T},
)::Nothing where {T<:half_quant}
    pdf[q, p, b] = one(T) - ε # Minus eps to counter + eps in stable log
    return nothing
end

const prior_pdf = Dict(
    "uniform" => uniform_pdf!,
    "gaussian" => gaussian_pdf!,
    "lognormal" => lognormal_pdf!,
    "ebm" => ebm_pdf!,
    "learnable_gaussian" => learnable_gaussian_pdf!,
)

end
