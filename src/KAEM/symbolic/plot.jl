module PlotKAN

export plot_ebm!

using CairoMakie, LaTeXStrings, Accessors, ConfParser
using NNlib: softmax
using MLDataDevices: cpu_device

using ..Utils
using ..FitSymbolic
using ..SymbolicFunctions

include("../ebm/quadrature.jl")
using .Quadrature: get_gausslegendre

include("func_lib.jl")
using .SymbolicLibrary

CairoMakie.activate!(type = "png")

function plot_ebm!(
        prior,
        ps,
        st_kan,
        st_lux,
        st_quad;
        plot_components = [(1, 1)],
        component_colours = [:red],
        file_loc = "figures/results/priors/",
        prior_type = "",
        fcn_type = "",
        show_formula = false,
        sym_fitter::Union{SymFitter, Nothing} = nothing,
    )
    mkpath(file_loc)

    z = first(
        get_gausslegendre(
            prior,
            st_kan,
            st_quad.init_nodes,
            st_quad.init_weights
        )
    )

    π_0 = prior.π_pdf(z, ps.dist.π_μ, ps.dist.π_σ)

    for i in 1:prior.depth
        @reset prior.fcns_qp[i].basis_function.S = prior.N_quad
    end
    @reset prior.s_size = prior.N_quad

    f = first(prior(ps, st_kan, st_lux, z))
    Z = first(prior.quad(prior, ps, st_kan, st_lux, st_quad))
    f = exp.(f) .* PermutedDimsArray(view(π_0, :, :, :), (3, 1, 2)) ./ sum(Z; dims = 3)

    # Fit symbolic functions if requested
    symbolic_fits = Dict()
    if show_formula && sym_fitter !== nothing
        for i in 1:prior.depth
            layer = prior.fcns_qp[i]
            ps_layer = cpu_device()(ps.fcn[symbol_map[i]])
            st_kan_layer = cpu_device()(st_kan[symbol_map[i]])
            fit_dict, α, β, w, b = sym_fitter(ps_layer, st_kan_layer, layer)

            sym_func = init_symbolic_function(
                layer.basis_function,
                layer.in_dim,
                layer.out_dim,
                st_kan_layer.grid[:, 1],
                st_kan_layer.grid[:, end],
                fit_dict,
                α, β, w, b
            )
            sym_ps = (α = α, β = β, w = w, b = b)
            symbolic_fits[i] = (sym_func, sym_ps)
        end
    end

    for (i, (q, p)) in enumerate(plot_components)
        fig = Figure(
            size = (800, 600),
            fontsize = 18,
            backgroundcolor = :white,
        )
        ax = Axis(
            fig[1, 1],
            title = "Prior component, q = $(q), p = $(p)",
            xlabel = L"z",
            ylabel = L"${\exp(f_{%$q,%$p}(z)) \cdot \pi_0(z)} \; / \; {\textbf{Z}_{%$q,%$p}}$",
            xgridvisible = true,
            ygridvisible = true,
            xgridstyle = :dash,
            ygridstyle = :dash,
            xgridcolor = (:gray, 0.3),
            ygridcolor = (:gray, 0.3),
        )

        z_vec = Vector{Float32}(z[p, :])
        f_vec = Vector{Float32}(f[q, p, :])
        π_0_vec = Vector{Float32}(π_0[p, :, 1])
        zero_vec = zeros(Float32, length(z_vec))

        band!(
            ax,
            z_vec,
            zero_vec,
            f_vec,
            color = (component_colours[i], 0.3),
            label = L"{\exp(f_{%$q,%$p}(z)) \cdot \pi_0(z)}",
        )
        band!(
            ax,
            z_vec,
            zero_vec,
            π_0_vec,
            color = (:gray, 0.2),
            label = L"\pi_0(z)",
        )

        lines!(
            ax, z_vec, f_vec,
            color = component_colours[i],
            linewidth = 2.5,
            label = nothing
        )
        lines!(
            ax, z_vec, π_0_vec,
            color = (:gray, 0.8),
            linewidth = 2.0,
            linestyle = :dash,
            label = nothing
        )

        axislegend(ax, position = :rt, framecolor = :gray, framevisible = true)

        if show_formula && !isempty(symbolic_fits)
            for layer_idx in 1:prior.depth
                if haskey(symbolic_fits, layer_idx)
                    sym_func, sym_ps = symbolic_fits[layer_idx]
                    if q <= sym_func.in_dim && p <= sym_func.out_dim
                        f_str = get_formula(sym_func, sym_ps, q, p)
                        f_latex = replace(f_str, "*" => " \\cdot ")
                        formula_latex = L"f_{%$q,%$p}(z) = %$f_latex"
                        text!(
                            ax,
                            0.03, 0.97 - 0.06 * (layer_idx - 1),
                            text = formula_latex,
                            align = (:left, :top),
                            space = :relative,
                            fontsize = 18,
                            color = :black,
                        )
                    end
                end
            end
        end

        save(
            file_loc * "/$(prior_type)_$(fcn_type)_$(q)_$(p).png",
            fig,
            px_per_unit = 2,
        )
    end
    return nothing
end

end
