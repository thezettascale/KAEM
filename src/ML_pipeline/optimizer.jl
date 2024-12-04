module optimization

export opt, create_opt

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

    LR = parse(Float32, retrieve(conf, "OPTIMIZER", "LR"))
    m = parse(Float32, retrieve(conf, "OPTIMIZER", "l-bfgs_memory"))
    c_1 = parse(Float32, retrieve(conf, "LINE_SEARCH", "c_1"))
    c_2 = parse(Float32, retrieve(conf, "LINE_SEARCH", "c_2"))
    ρ = parse(Float32, retrieve(conf, "LINE_SEARCH", "rho"))
    ls_type = retrieve(conf, "LINE_SEARCH", "type")
    opt_type = retrieve(conf, "OPTIMIZER", "type")
    
    linesearch = Dict(
        "strongwolfe" => LineSearches.StrongWolfe{Float32}(c_1=c_1, c_2=c_2, ρ=ρ),
        "backtrack" => LineSearches.BackTracking{Float32}(c_1=c_1, ρ_hi=ρ, ρ_lo=1f-1, maxstep=Inf32),
        "hagerzhang" => LineSearches.HagerZhang{Float32}(),
        "morethuente" => LineSearches.MoreThuente{Float32}(f_tol=0f0, gtol=0f0, x_tol=0f0),
    )[ls_type]

    linesearch = (a...) -> linesearch(a...) 

    optimiser_map = Dict(
        "bfgs" => BFGS(alphaguess=LineSearches.InitialHagerZhang{Float32}(α0=LR), linesearch=linesearch),
        "l-bfgs" => LBFGS(alphaguess=LineSearches.InitialHagerZhang{Float32}(α0=LR), m=o.m, linesearch=linesearch),
        "cg" => ConjugateGradient(alphaguess=LineSearches.InitialHagerZhang{Float32}(α0=LR), linesearch=linesearch),
        "gd" => GradientDescent(alphaguess=LineSearches.InitialHagerZhang{Float32}(α0=LR), linesearch=linesearch),
        "newton" => Newton(alphaguess=LineSearches.InitialHagerZhang{Float32}(α0=LR), linesearch=linesearch),
        "interior-point" => IPNewton(linesearch=linesearch),
        "neldermead" => NelderMead(),
        "adam" => ADAM(LR),
        "adamw" => ADAMW(LR),
        "sgd" => Descent(LR),
        "rms" => RMSProp(LR, 9f-1, 1f-8),
    )

    init_fcn = () -> optimiser_map[opt_type]

    return opt(init_fcn)
end

end