module AdamOptimizer

export ManualAdam

using Optimisers
using Optimisers: @..

struct ManualAdam{T <: Real} <: Optimisers.AbstractRule
    eta::T
    beta::Tuple{T, T}
    decay::T
    epsilon::T
    couple::Bool
end

ManualAdam(
    eta::T,
    beta::Tuple{T, T} = (T(0.9), T(0.999)),
    decay::T = zero(T),
    epsilon::T = T(1.0e-8);
    couple::Bool = true,
) where {T <: Real} = ManualAdam{T}(eta, beta, decay, epsilon, couple)

Optimisers.init(o::ManualAdam, x::AbstractArray) = (m = zero(x), v = zero(x), t = 0)

function Optimisers.apply!(o::ManualAdam, st, x::AbstractArray, g)
    β1, β2 = o.beta
    t = st.t + 1

    m = st.m
    v = st.v

    if !o.couple && (o.decay > 0.0f0)
        g = g .+ o.decay .* x
    end

    # Momentum update
    @.. m = β1 * m + (1 - β1) * g
    @.. v = β2 * v + (1 - β2) * (g * g)

    # Bias correction
    β1t = β1^t
    β2t = β2^t
    mhat = m ./ (1 - β1t)
    vhat = v ./ (1 - β2t)

    # Adam step
    step = o.eta .* mhat ./ (sqrt.(vhat) .+ o.epsilon)

    # Weight decay contribution
    if o.couple && (o.decay > 0.0f0)
        step = step .+ (o.eta * o.decay) .* x
    end

    return (m = m, v = v, t = t), step
end

end
