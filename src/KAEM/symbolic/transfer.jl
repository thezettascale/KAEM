module Transfer

export SymbolicTransfer, transfer_to_symbolic

using Accessors, ConfParser, Lux, Random, Reactant
using MLDataDevices

using ..Utils
using ..KAEM_model
using ..KAEM_model.SymbolicFunctions
using ..FitSymbolic

include("func_lib.jl")
using .SymbolicLibrary

struct SymbolicTransfer
    transfer_ebm::Bool
    transfer_gen::Bool
    sym_fitter_ebm::Union{SymFitter, Nothing}
    sym_fitter_gen::Union{SymFitter, Nothing}
end

function SymbolicTransfer(
        conf::ConfParse,
        CNN_bool::Bool,
        SEQ_bool::Bool;
        symbolic_lib_prior = SYMB_LIB,
        symbolic_lib_llhood = SYMB_LIB
    )
    transfer_ebm = parse(Bool, retrieve(conf, "SYMBOLIC_REG", "fit_symbolic_prior"))
    transfer_gen = parse(Bool, retrieve(conf, "SYMBOLIC_REG", "fit_symbolic_llhood"))
    transfer_gen = transfer_gen && !CNN_bool && !SEQ_bool

    sf_ebm = (
        transfer_ebm ?
            SymFitter(conf; symbolic_lib = symbolic_lib_prior) :
            nothing
    )
    sf_gen = (
        transfer_gen ?
            SymFitter(conf; symbolic_lib = symbolic_lib_llhood) :
            nothing
    )

    return SymbolicTransfer(
        transfer_ebm,
        transfer_gen,
        sf_ebm,
        sf_gen
    )
end

function (t::SymbolicTransfer)(
        model,
        ps,
        st_kan;
        rng = Random.MersenneTwister(1)
    )
    model_copy = model

    if t.transfer_ebm
        prior_copy = model_copy.prior

        for i in 1:prior_copy.depth
            layer = prior_copy.fcns_qp[i]
            I, O = layer.in_dim, layer.out_dim

            ps_layer = ps.ebm.fcn[symbol_map[i]]
            st_kan_layer = st_kan.ebm[symbol_map[i]]
            layer = prior_copy.fcns_qp[i]

            fit_dict, α, β, w, b = t.sym_fitter_ebm(
                ps_layer,
                st_kan_layer,
                layer;
                rng = rng
            )

            sym_func = init_symbolic_function(
                layer.basis_function,
                I,
                O,
                st_kan_layer.grid[:, 1],
                st_kan_layer.grid[:, end],
                fit_dict,
                α,
                β,
                w,
                b,
            )

            @reset model.prior.fcns_qp[i] = sym_func
        end
    end

    if t.transfer_gen
        gen_copy = model_copy.lkhood.generator

        for i in 1:gen_copy.depth
            layer = gen_copy.Φ_fcns[i]
            I, O = layer.in_dim, layer.out_dim

            ps_layer = ps.gen.fcn[symbol_map[i]]
            st_kan_layer = st_kan.gen[symbol_map[i]]
            layer = gen_copy.Φ_fcns[i]

            fit_dict, α, β, w, b = t.sym_fitter_gen(
                ps_layer,
                st_kan_layer,
                layer;
                rng = rng
            )

            sym_func = init_symbolic_function(
                layer.basis_function,
                I,
                O,
                st_kan_layer.grid[:, 1],
                st_kan_layer.grid[:, end],
                fit_dict,
                α,
                β,
                w,
                b,
            )

            @reset gen_copy.Φ_fcns[i] = sym_func
        end

        @reset model.lkhood.generator = gen_copy
    end

    return model
end

end
