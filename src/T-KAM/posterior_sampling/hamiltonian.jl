module HamiltonianDynamics

export position_update_4d, position_update_3d, momentum_update_4d, momentum_update_3d

using CUDA, KernelAbstractions, Tullio, Lux, LuxCUDA

include("../../utils.jl")
using .Utils: full_quant

function position_update_4d(
    z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},
    ∇z::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η::AbstractArray{full_quant},
    )
    @tullio y[q,p,s,t] := momentum[q,p,s,t] + (η[s,t]/2) * ∇z[q,p,s,t] / M[q,p,s,t]
    @tullio ẑ[q,p,s,t] := z[q,p,s,t] + η[s,t] * y[q,p,s,t] / M[q,p,s,t]
    return y, ẑ
end

function position_update_3d(
    z::AbstractArray{full_quant},
    momentum::AbstractArray{full_quant},
    ∇z::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η::AbstractArray{full_quant},
    )
    @tullio y[q,p,s] := momentum[q,p,s] + (η[s]/2) * ∇z[q,p,s] / M[q,p,s]
    @tullio ẑ[q,p,s] := z[q,p,s] + η[s] * y[q,p,s] / M[q,p,s]
    return y, ẑ
end

function momentum_update_4d(
    y::AbstractArray{full_quant},
    ∇ẑ::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η::AbstractArray{full_quant},
    )
    @tullio y_out[q,p,s,t] := y[q,p,s,t] + (η[s,t]/2) * ∇ẑ[q,p,s,t] / M[q,p,s,t]
    return y_out
end

function momentum_update_3d(
    y::AbstractArray{full_quant},
    ∇ẑ::AbstractArray{full_quant},
    M::AbstractArray{full_quant},
    η::AbstractArray{full_quant}
    )
    @tullio y_out[q,p,s] := y[q,p,s] + (η[s]/2) * ∇ẑ[q,p,s] / M[q,p,s]
    return y_out
end

end 
