module ModelGridUpdating

export GridUpdater

using Accessors, Lux, ComponentArrays, ConfParser

using ..Utils
using ..KAEM_model
using ..KAEM_model.UnivariateFunctions
using ..KAEM_model.EBM_Model
using ..KAEM_model.Quadrature: get_gausslegendre

include("kan/grid_updating.jl")
using .GridUpdating

struct GridUpdater
    model
    update_frequency
    update_decay
    update_prior_grid
    update_llhood_grid
    nogrid_prior
end

function GridUpdater(model, conf::ConfParse)
    grid_update_frequency =
        parse(Int, retrieve(conf, "GRID_UPDATING", "grid_update_frequency"))
    grid_update_decay =
        parse(Float32, retrieve(conf, "GRID_UPDATING", "grid_update_decay"))
    update_prior_grid = parse(Bool, retrieve(conf, "GRID_UPDATING", "update_prior_grid"))
    update_llhood_grid = parse(Bool, retrieve(conf, "GRID_UPDATING", "update_llhood_grid"))

    prior_func = retrieve(conf, "EbmModel", "spline_function")
    gen_func = retrieve(conf, "GeneratorModel", "spline_function")

    nogrid_prior = prior_func == "FFT" || prior_func == "Cheby"
    nogrid_gen = gen_func == "FFT" || gen_func == "Cheby"

    return GridUpdater(
        model,
        grid_update_frequency,
        grid_update_decay,
        update_prior_grid,
        update_llhood_grid && !model.lkhood.CNN && !model.lkhood.SEQ && !nogrid_gen,
        nogrid_prior,
    )
end

function (gu::GridUpdater)(
        x,
        ps,
        st_kan,
        st_lux,
        train_idx,
        st_rng,
    )
    """
    Update the grid using samples from the prior.

    Args:
        x: Data samples.
        ps: The parameters of the model.
        st_kan: The states of the KAN model.
        st_lux: The states of the Lux model.
        temps: The temperatures for thermodynamic models.
        rng: The random number generator.

    Returns:
        The updated params.
        The updated KAN states.
        The updated Lux states. 
    """

    model = gu.model
    z = nothing
    if gu.update_prior_grid

        if model.N_t > 1
            temps = collect(Float32, [(k / model.N_t)^compute_p(model, train_idx) for k in 1:model.N_t])
            z = first(
                model.posterior_sampler(
                    ps,
                    st_kan,
                    st_lux,
                    x,
                    st_rng;
                    temps = temps,
                ),
            )[
                :,
                :,
                :,
                end,
            ]
        elseif model.prior.bool_config.ula || model.MALA
            z = first(model.posterior_sampler(ps, st_kan, st_lux, x, st_rng))[
                :,
                :,
                :,
                1,
            ]
        else
            z = first(
                model.sample_prior(
                    model,
                    ps,
                    st_kan,
                    st_lux,
                    st_rng,
                )
            )
            # z = first(model.posterior_sampler(ps, st_kan, st_lux, x, st_rng))[
            #     :,
            #     :,
            #     :,
            #     1,
            # ] # For domain updating: use ULA to explore beyond prior init domain.
        end

        # Must update domain for inverse transform sampling
        ula_bool = model.prior.bool_config.ula || model.MALA || model.N_t > 1
        if (ula_bool && gu.nogrid_prior)
            red_dim = model.prior.bool_config.mixture_model ? (2, 3) : (1, 3)
            min_z = dropdims(minimum(z; dims = red_dim); dims = red_dim)
            max_z = dropdims(maximum(z; dims = red_dim); dims = red_dim)

            # Ensure min_z < max_z
            order_bool = min_z .< max_z
            min_z = ifelse.(order_bool, min_z, max_z)
            max_z = ifelse.(order_bool, max_z, min_z)

            # Expand bounds slightly
            low, high = zero(min_z) .+ 0.95f0, zero(max_z) .+ 1.05f0
            lo_bound = ifelse.(min_z .< 0, high, low)
            hi_bound = ifelse.(max_z .< 0, low, high)

            @reset st_kan.ebm[:a].min = lo_bound .* min_z
            @reset st_kan.ebm[:a].max = hi_bound .* max_z
        end

        if !gu.nogrid_prior
            prior_copy = model.prior
            Q, P, B = prior_copy.q_size, prior_copy.p_size, model.batch_size

            if !prior_copy.bool_config.ula && !prior_copy.bool_config.mixture_model
                for i in 1:prior_copy.depth
                    @reset prior_copy.fcns_qp[i].basis_function.S = Q * B
                end
                @reset prior_copy.s_size = Q * B
                z = reshape(z, P, Q * B)
            else
                z = dropdims(z; dims = 2)
            end

            mid_size = prior_copy.bool_config.mixture_model ? P : Q
            outer_dim = prior_copy.bool_config.mixture_model ? Q * B : Q * P * B

            for i in 1:prior_copy.depth
                if prior_copy.bool_config.layernorm && i != 1
                    z, st_ebm = Lux.apply(
                        prior_copy.layernorms[i - 1],
                        z,
                        ps.ebm.layernorm[symbol_map[i]],
                        st_lux.ebm[symbol_map[i]],
                    )
                    @reset st_lux.ebm[symbol_map[i]] = st_ebm
                end

                new_grid, new_coef = update_fcn_grid(
                    prior_copy.fcns_qp[i],
                    ps.ebm.fcn[symbol_map[i]],
                    st_kan.ebm[symbol_map[i]],
                    z,
                )
                @views ps[:ebm][:fcn][symbol_map[i]][:coef] .= new_coef
                @reset st_kan.ebm[symbol_map[i]].grid = new_grid

                if prior_copy.fcns_qp[i].spline_string == "RBF"
                    scale = (maximum(new_grid) - minimum(new_grid)) /
                        (size(new_grid, 2) - 1) |> Lux.f32

                    @reset st_kan.ebm[symbol_map[i]].scale = scale .+ zero(st_kan.ebm[symbol_map[i]].scale)
                end

                z = Lux.apply(
                    prior_copy.fcns_qp[i],
                    z,
                    ps.ebm.fcn[symbol_map[i]],
                    st_kan.ebm[symbol_map[i]],
                )
                z =
                    (i == 1 && !prior_copy.bool_config.ula) ? reshape(z, mid_size, outer_dim) :
                    dropdims(sum(z, dims = 1); dims = 1)
            end
        end
    end

    new_nodes, new_weights = get_gausslegendre(
        model.prior,
        st_kan.ebm,
        st_kan.quad.init_nodes,
        st_kan.quad.init_weights
    )
    @reset st_kan.quad.nodes = new_nodes
    @reset st_kan.quad.weights = new_weights

    # Only update if KAN-type generator requires
    if gu.update_llhood_grid
        if model.N_t > 1
            temps = collect(Float32, [(k / model.N_t)^compute_p(model, train_idx) for k in 1:model.N_t])
            z = first(
                model.posterior_sampler(
                    ps,
                    st_kan,
                    st_lux,
                    x,
                    st_rng;
                    temps = temps,
                ),
            )[
                :,
                :,
                :,
                end,
            ]
        elseif model.prior.bool_config.ula || model.MALA
            z = first(model.posterior_sampler(ps, st_kan, st_lux, x, st_rng))[
                :,
                :,
                :,
                1,
            ]
        else
            z = first(
                model.sample_prior(model, ps, st_kan, st_lux, st_rng)
            )
            # z = first(model.posterior_sampler(ps, st_kan, st_lux, x, st_rng))[
            #     :,
            #     :,
            #     :,
            #     1,
            # ] # For domain updating: use ULA to explore beyond prior init domain.
        end

        z = dropdims(sum(z; dims = 2); dims = 2)

        for i in 1:model.lkhood.generator.depth
            if model.lkhood.generator.bool_config.layernorm
                z, st_gen = Lux.apply(
                    model.lkhood.generator.layernorms[i],
                    z,
                    ps.gen.layernorm[symbol_map[i]],
                    st_lux.gen[symbol_map[i]],
                )
                @reset st_lux.gen[symbol_map[i]] = st_gen
            end

            if !(
                    model.lkhood.generator.Φ_fcns[i].spline_string == "FFT" ||
                        model.lkhood.generator.Φ_fcns[i].spline_string == "Cheby"
                )
                new_grid, new_coef = update_fcn_grid(
                    model.lkhood.generator.Φ_fcns[i],
                    ps.gen.fcn[symbol_map[i]],
                    st_kan.gen[symbol_map[i]],
                    z,
                )
                @views ps[:gen][:fcn][symbol_map[i]][:coef] .= new_coef
                @reset st_kan.gen[symbol_map[i]].grid = new_grid

                if model.lkhood.generator.Φ_fcns[i].spline_string == "RBF"
                    scale = (maximum(new_grid) - minimum(new_grid)) /
                        (size(new_grid, 2) - 1) |> Lux.f32

                    @reset st_kan.gen[symbol_map[i]].scale = scale .+ zero(st_kan.gen[symbol_map[i]].scale)
                end
            end

            z = Lux.apply(
                model.lkhood.generator.Φ_fcns[i],
                z,
                ps.gen.fcn[symbol_map[i]],
                st_kan.gen[symbol_map[i]],
            )
            z = dropdims(sum(z, dims = 1); dims = 1)
        end
    end

    return ps, st_kan, st_lux
end

end
