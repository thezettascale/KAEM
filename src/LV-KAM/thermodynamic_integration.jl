module ThermodynamicIntegration

export Thermodynamic_LV_KAM

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, Accessors
using Flux: DataLoader

include("mixture_prior.jl")
include("MoE_likelihood.jl")
using .ebm_mix_prior
using .MoE_likelihood

struct Thermodynamic_LV_KAM <: Lux.AbstractLuxLayer
    prior::mix_prior
    lkhood::MoE_lkhood
    train_loader::DataLoader
    test_loader::DataLoader
    update_prior_grid::Bool
    update_llhood_grid::Bool
    grid_update_decay::Float32
    grid_updates_samples::Int
    MC_samples::Int
    verbose::Bool
    temperatures::AbstractArray{Float32}
end

function Lux.initialparameters(rng::AbstractRNG, model::Thermodynamic_LV_KAM)
    return (ebm = Lux.initialparameters(rng, model.prior), gen = Lux.initialparameters(rng, model.lkhood))
end

function Lux.initialstates(rng::AbstractRNG, model::Thermodynamic_LV_KAM)
    return (ebm = Lux.initialstates(rng, model.prior), gen = Lux.initialstates(rng, model.lkhood))
end



end