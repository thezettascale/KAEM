module Reg

export Regularizer

using Statistics, ConfParser, Accessors, Lux

using ..Utils
using ..KAEM_model
using ..KAEM_model.UnivariateFunctions
using ..KAEM_model.EBM_Model

struct Regularizer
    ε
    th
    μ
    λ_total
    λ_l1
    λ_entropy
    λ_coef
    λ_coefdiff
    reg_ebm
    reg_gen
end

function Regularizer(conf::ConfParse, CNN_bool::Bool, SEQ_bool::Bool)
    ε = parse(Float32, retrieve(conf, "TRAINING", "eps"))
    th = parse(Float32, retrieve(conf, "SYMBOLIC_REG", "nonlin_threshold"))
    μ = parse(Float32, retrieve(conf, "SYMBOLIC_REG", "reg_factor"))
    λ_total = parse(Float32, retrieve(conf, "SYMBOLIC_REG", "λ_total"))
    λ_l1 = parse(Float32, retrieve(conf, "SYMBOLIC_REG", "λ_l1"))
    λ_entropy = parse(Float32, retrieve(conf, "SYMBOLIC_REG", "λ_entropy"))
    λ_coef = parse(Float32, retrieve(conf, "SYMBOLIC_REG", "λ_coef"))
    λ_coefdiff = parse(Float32, retrieve(conf, "SYMBOLIC_REG", "λ_coefdiff"))
    reg_ebm = parse(Bool, retrieve(conf, "SYMBOLIC_REG", "regularize_prior"))
    reg_gen = parse(Bool, retrieve(conf, "SYMBOLIC_REG", "regularize_gen"))
    return Regularizer(
        ε,
        th,
        μ,
        λ_total,
        λ_l1,
        λ_entropy,
        λ_coef,
        λ_coefdiff,
        reg_ebm,
        reg_gen && !CNN_bool && !SEQ_bool,
    )
end

function nonlin_term(
        x;
        th = 1.0f-16,
        factor = 1.0f0
    )
    term1 = (x .< th) |> Lux.f32
    term2 = (x .>= th) |> Lux.f32
    return term1 .* x .* factor .+
        term2 .* (x .+ (factor - 1) .* th)
end

function regularize(
        r,
        l,
        x,
        ps,
        st
    )
    x_expanded = PermutedDimsArray(view(x, :, :, :), (1, 3, 2))
    in_range = std(x_expanded; dims = 3) .+ r.ε

    y = l(x, ps, st)
    out_range = std(y; dims = 3)

    scale_factor = dropdims(out_range ./ in_range; dims = 3)
    scale_factor ./= sum(scale_factor)

    l1 = sum(nonlin_term(scale_factor; th = r.th, factor = r.μ))
    entropy = -1.0f0 * sum(scale_factor .* log.(2, scale_factor .+ r.ε))

    coef = ps.coef
    coef_l1 = sum(mean(abs.(coef); dims = 1))
    coef_diff_l1 = sum(mean(abs.(diff(coef; dims = 2)); dims = 1))

    return (
        r.λ_l1 * l1 +
            r.λ_entropy * entropy +
            r.λ_coef * coef_l1 +
            r.λ_coefdiff * coef_diff_l1
    )
end

function (r::Regularizer)(
        z,
        model,
        ps,
        st_kan,
        st_ebm,
        st_gen,
    )
    reg = 0.0f0
    z_gen = z
    if r.reg_ebm

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
                z, st_lux_ebm = Lux.apply(
                    prior_copy.layernorms[i - 1],
                    z,
                    ps.ebm.layernorm[symbol_map[i]],
                    st_ebm[symbol_map[i]],
                )
                @reset st_ebm[symbol_map[i]] = st_lux_ebm
            end

            reg += regularize(
                r,
                prior_copy.fcns_qp[i],
                z,
                ps.ebm.fcn[symbol_map[i]],
                st_kan.ebm[symbol_map[i]],
            )

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

    if r.reg_gen

        z = dropdims(sum(z_gen; dims = 2); dims = 2)

        for i in 1:model.lkhood.generator.depth
            if model.lkhood.generator.bool_config.layernorm
                z, st_lux_gen = Lux.apply(
                    model.lkhood.generator.layernorms[i],
                    z,
                    ps.gen.layernorm[symbol_map[i]],
                    st_gen[symbol_map[i]],
                )
                @reset st_gen[symbol_map[i]] = st_lux_gen
            end

            reg += regularize(
                r,
                model.lkhood.generator.Φ_fcns[i],
                z,
                ps.gen.fcn[symbol_map[i]],
                st_kan.gen[symbol_map[i]],
            )

            z = Lux.apply(
                model.lkhood.generator.Φ_fcns[i],
                z,
                ps.gen.fcn[symbol_map[i]],
                st_kan.gen[symbol_map[i]],
            )
            z = dropdims(sum(z, dims = 1); dims = 1)
        end
    end

    return reg * r.λ_total, st_ebm, st_gen
end

end
