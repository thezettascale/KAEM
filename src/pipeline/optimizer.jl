module optimization

export opt, create_opt

include("../utils.jl")
using .Utils: full_quant

using Lux, OptimizationOptimJL, LineSearches, OptimizationOptimisers, ConfParser

struct opt
    init_optimizer::Function
end

function create_opt(conf::ConfParse)
    """
    Create an optimizer from a configuration file.

    Args:
        conf: ConfParse object

    Returns:
        opt: opt object, which initializes the optimizer when called
    """

    LR = parse(full_quant, retrieve(conf, "OPTIMIZER", "learning_rate"))
    m = parse(Int, retrieve(conf, "OPTIMIZER", "l-bfgs_memory"))
    c_1 = parse(full_quant, retrieve(conf, "LINE_SEARCH", "c_1"))
    c_2 = parse(full_quant, retrieve(conf, "LINE_SEARCH", "c_2"))
    ρ = parse(full_quant, retrieve(conf, "LINE_SEARCH", "rho"))
    ls_type = retrieve(conf, "LINE_SEARCH", "type")
    opt_type = retrieve(conf, "OPTIMIZER", "type")

    linesearch = Dict(
        "strongwolfe" =>
            LineSearches.StrongWolfe{full_quant}(c_1 = c_1, c_2 = c_2, ρ = ρ),
        "backtrack" => LineSearches.BackTracking{full_quant}(
            c_1 = c_1,
            ρ_hi = ρ,
            ρ_lo = full_quant(0.1),
            maxstep = Inf32,
        ),
        "hagerzhang" => LineSearches.HagerZhang{full_quant}(),
        "morethuente" => LineSearches.MoreThuente{full_quant}(
            f_tol = zero(full_quant),
            gtol = zero(full_quant),
            x_tol = zero(full_quant),
        ),
    )[ls_type]

    linesearch = (a...) -> linesearch(a...)

    optimiser_map = Dict(
        "bfgs" => BFGS(
            alphaguess = LineSearches.InitialHagerZhang{full_quant}(α0 = LR),
            linesearch = linesearch,
        ),
        "l-bfgs" => LBFGS(
            alphaguess = LineSearches.InitialHagerZhang{full_quant}(α0 = LR),
            m = m,
            linesearch = linesearch,
        ),
        "cg" => ConjugateGradient(
            alphaguess = LineSearches.InitialHagerZhang{full_quant}(α0 = LR),
            linesearch = linesearch,
        ),
        "gd" => GradientDescent(
            alphaguess = LineSearches.InitialHagerZhang{full_quant}(α0 = LR),
            linesearch = linesearch,
        ),
        "newton" => Newton(
            alphaguess = LineSearches.InitialHagerZhang{full_quant}(α0 = LR),
            linesearch = linesearch,
        ),
        "interior-point" => IPNewton(linesearch = linesearch),
        "neldermead" => NelderMead(),
        "adam" => ADAM(LR),
        "adamw" => ADAMW(LR),
        "sgd" => Descent(LR),
        "rms" => RMSProp(LR, 9.0f-1, full_quant(1e-8)),
    )

    init_fcn = () -> optimiser_map[opt_type]

    return opt(init_fcn)
end

end
