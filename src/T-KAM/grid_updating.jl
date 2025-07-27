
module ModelGridUpdating

export update_model_grid

using CUDA,
    KernelAbstractions,
    Accessors,
    ComponentArrays,
    Lux,
    NNlib,
    LinearAlgebra,
    Random,
    LuxCUDA

using ..Utils
using ..T_KAM_model
using ..T_KAM_model.UnivariateFunctions

include("kan/grid_updating.jl")
using .GridUpdating

function update_model_grid(
    model::T_KAM{T,U},
    x::AbstractArray{T},
    ps::ComponentArray{T},
    st_kan::ComponentArray{T},
    st_lux::NamedTuple;
    temps::AbstractArray{T} = [one(T)],
    rng::AbstractRNG = Random.default_rng(),
)::Tuple{
    Any,
    ComponentArray{T},
    ComponentArray{T},
    NamedTuple,
} where {T<:half_quant,U<:full_quant}
    """
    Update the grid of the likelihood model using samples from the prior.

    Args:
        model: The model.
        x: Data samples.
        ps: The parameters of the model.
        st_kan: The states of the KAN model.
        st_lux: The states of the Lux model.
        temps: The temperatures for thermodynamic models.
        rng: The random number generator.

    Returns:
        The updated model.
        The updated params.
        The updated KAN states.
        The updated Lux states. 
    """

    z = nothing
    sampled_bool = false
    if model.update_prior_grid

        if model.N_t > 1
            z = first(
                model.posterior_sampler(
                    model,
                    ps,
                    st_kan,
                    st_lux,
                    x;
                    temps = temps[2:end],
                    rng = rng,
                ),
            )
        elseif model.prior.ula || model.MALA
            z = first(model.posterior_sampler(model, ps, st_kan, st_lux, x; rng = rng))
        else
            z = first(
                model.sample_prior(
                    model,
                    model.grid_updates_samples,
                    ps,
                    st_kan,
                    st_lux,
                    rng,
                ),
            )
        end

        sampled_bool = true

        # If Cheby or FFT, need to update domain for inverse transform sampling
        if model.prior.fcns_qp[1].spline_string == "FFT" ||
           model.prior.fcns_qp[1].spline_string == "Cheby"
            if (model.MALA || model.N_t > 1 || model.prior.ula)
                new_domain = (minimum(z), maximum(z))
                @reset model.prior.fcns_qp[1].grid_range = new_domain
            end

            # Otherwise use KAN grid updating
        else
            Q, P = (
                (model.prior.ula || model.prior.mixture_model) ? reverse(size(z)[1:2]) :
                size(z)[1:2]
            )
            z = reshape(z, P, Q, :)
            B = size(z, 3)
            z = reshape(z, P, Q*B)

            for i = 1:model.prior.depth
                new_grid, new_coef = update_fcn_grid(
                    model.prior.fcns_qp[i],
                    ps.ebm.fcn[symbol_map[i]],
                    st_kan.ebm[symbol_map[i]],
                    z,
                )
                @reset ps.ebm.fcn[symbol_map[i]].coef = new_coef
                @reset st_kan.ebm[symbol_map[i]].grid = new_grid

                scale = (maximum(new_grid) - minimum(new_grid)) / (size(new_grid, 2) - 1)
                model.prior.fcns_qp[i].spline_string == "RBF" &&
                    @reset model.prior.fcns_qp[i].basis_function.scale = scale

                z = Lux.apply(
                    model.prior.fcns_qp[i],
                    z,
                    ps.ebm.fcn[symbol_map[i]],
                    st_kan.ebm[symbol_map[i]],
                )
                z =
                    i == 1 ? reshape(z, size(z, 2), :) :
                    dropdims(sum(z, dims = 1); dims = 1)

                if model.prior.layernorm_bool && i < model.prior.depth
                    z, st_ebm = Lux.apply(
                        model.prior.layernorms[i],
                        z,
                        ps.ebm.layernorm[symbol_map[i]],
                        st_lux.ebm.layernorm[symbol_map[i]],
                    )
                    @reset st_lux.ebm.layernorm[symbol_map[i]] = st_ebm
                end
            end
        end
    end

    # Only update if KAN-type generator requires
    (!model.update_llhood_grid || model.lkhood.CNN || model.lkhood.SEQ) &&
        return model, T.(ps), st_kan, st_lux

    if !sampled_bool
        if model.N_t > 1
            z = first(
                model.posterior_sampler(
                    model,
                    ps,
                    st_kan,
                    st_lux,
                    x;
                    temps = temps[2:end],
                    rng = rng,
                ),
            )
        elseif model.prior.ula || model.MALA
            z = first(model.posterior_sampler(model, ps, st_kan, st_lux, x; rng = rng))
        else
            z = first(
                model.sample_prior(
                    model,
                    model.grid_updates_samples,
                    ps,
                    st_kan,
                    st_lux,
                    rng,
                ),
            )
        end
    end

    z = dropdims(sum(reshape(z, size(z, 1), size(z, 2), :); dims = 2); dims = 2)

    for i = 1:model.lkhood.generator.depth
        new_grid, new_coef = update_fcn_grid(
            model.lkhood.generator.Φ_fcns[i],
            ps.gen.fcn[symbol_map[i]],
            st_kan.gen[symbol_map[i]],
            z,
        )
        @reset ps.gen.fcn[symbol_map[i]].coef = new_coef
        @reset st_kan.gen[symbol_map[i]].grid = new_grid

        scale = (maximum(new_grid) - minimum(new_grid)) / (size(new_grid, 2) - 1)
        model.lkhood.generator.Φ_fcns[i].spline_string == "RBF" &&
            @reset model.lkhood.generator.Φ_fcns[i].basis_function.scale = scale

        z = Lux.apply(
            model.lkhood.generator.Φ_fcns[i],
            z,
            ps.gen.fcn[symbol_map[i]],
            st_kan.gen[symbol_map[i]],
        )
        z = dropdims(sum(z, dims = 1); dims = 1)

        if model.lkhood.generator.layernorm_bool && i < model.lkhood.generator.depth
            z, st_gen = Lux.apply(
                model.lkhood.generator.layernorms[i],
                z,
                ps.gen.layernorm[symbol_map[i]],
                st_lux.gen.layernorm[symbol_map[i]],
            )
            @reset st_lux.gen.layernorm[symbol_map[i]] = st_gen
        end
    end

    return model, T.(ps), st_kan, st_lux
end

end
