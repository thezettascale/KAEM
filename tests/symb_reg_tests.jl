using Test, Random, LinearAlgebra, Statistics, ComponentArrays, ConfParser, Lux

ENV["GPU"] = false # Don't change

include("../src/utils.jl")
using .Utils

include("../src/KAEM/symbolic/func_lib.jl")
using .SymbolicLibrary

include("../src/KAEM/symbolic/fit.jl")
using .FitSymbolic

include("../src/KAEM/KAEM.jl")
using .KAEM_model
using .KAEM_model: AbstractBasis

include("../src/KAEM/model_setup.jl")
using .ModelSetup

include("../src/pipeline/optimizer.jl")
using .optimization

include("../src/KAEM/symbolic/reg.jl")
using .Reg

include("../src/KAEM/kan/univariate_functions.jl")
using .UnivariateFunctions: RBF_basis

include("../src/KAEM/symbolic/symbolic_func.jl")
using .SymbolicFunctions

include("../src/KAEM/symbolic/transfer.jl")
using .Transfer

include("../src/KAEM/kan/spline_bases.jl")
using .spline_functions

Random.seed!(42)
conf = ConfParse("tests/test_conf.ini")
parse_conf!(conf)

b_size = parse(Int, retrieve(conf, "SYMBOLIC_REG", "num_points_fitting"))
test_symb_lib = Dict(
    "x" => SYMB_LIB["x"],
    "x^2" => SYMB_LIB["x^2"],
    "sin" => SYMB_LIB["sin"],
    "exp" => SYMB_LIB["exp"]
)

function test_symbolic_functions()
    x = rand(Float32, 10)
    x_pos = abs.(x) .+ 1.0f-3

    # Test basic functions
    @test all(SYMB_LIB["x"][1](x) .== x)
    @test all(SYMB_LIB["x^2"][1](x) .== x .^ 2)
    @test all(SYMB_LIB["x^3"][1](x) .== x .^ 3)
    @test all(SYMB_LIB["abs"][1](x) .== abs.(x))
    @test all(SYMB_LIB["sin"][1](x) .== sin.(x))
    @test all(SYMB_LIB["cos"][1](x) .== cos.(x))

    y_th = 1.0f2
    result = SYMB_LIB["1/x"][3]((x_pos), (y_th))
    @test !any(isnan.(result[2]))
    @test !any(isinf.(result[2]))

    result = SYMB_LIB["sqrt"][3]((x_pos), (y_th))
    @test !any(isnan.(result[2]))
    @test !any(isinf.(result[2]))

    result = SYMB_LIB["log"][3]((x_pos), (y_th))
    @test !any(isnan.(result[2]))

    return @test length(SYMB_LIB) > 0
end

function test_ols_wb()
    Random.seed!(42)
    n, f = 100, 200
    w_true = 2.0f0
    b_true = 1.0f0
    z = randn(Float32, n, f)
    y = w_true .* z .+ b_true .+ 0.1f0 .* randn(Float32, n, f)

    w, b = FitSymbolic.ols_wb(z, y)

    @test all(abs.(w .- w_true) .< 0.5f0)
    @test all(abs.(b .- b_true) .< 0.5f0)
    @test !any(isnan.(w))
    @test !any(isnan.(b))

    z_zero = zeros(Float32, n, f)
    w_zero, b_zero = FitSymbolic.ols_wb(z_zero, y)
    @test all(w_zero .≈ 0.0f0)
    @test !any(isnan.(b_zero))
    return @test true
end

function test_fit_affine()
    Random.seed!(42)
    I, O, N = 3, 2, 50
    x = randn(Float32, I, O, N)

    func = x -> x
    α_true = ones(Float32, I, O)
    β_true = zeros(Float32, I, O)
    y = func.(α_true .* x .+ β_true)

    R2, α, β = FitSymbolic.fit_affine(x, y, func, I, O; max_iters = 50)

    @test size(R2) == (I, O)
    @test size(α) == (I, O)
    @test size(β) == (I, O)
    @test !any(isnan.(R2))
    @test !any(isnan.(α))
    @test !any(isnan.(β))
    @test all(R2 .> -1.0f0)

    func_sq = x -> x .^ 2
    y_sq = func_sq.(α_true .* x .+ β_true)
    R2_sq, α_sq, β_sq = FitSymbolic.fit_affine(
        x,
        y_sq,
        func_sq,
        I,
        O;
        max_iters = 50
    )

    @test size(R2_sq) == (I, O)
    @test !any(isnan.(R2_sq))

    return @test true
end

function test_fit_symbolic()
    Random.seed!(42)
    I, O, N = 2, 2, 30
    x = randn(Float32, I, O, N)

    func = x -> x
    R2, α, β, w, b = FitSymbolic.fit_symbolic(
        x,
        x,
        func,
        I,
        O;
        max_iters = 30
    )

    @test size(R2) == (I, O)
    @test size(α) == (I, O)
    @test size(β) == (I, O)
    @test size(w) == (I, O)
    @test size(b) == (I, O)
    @test !any(isnan.(R2))
    @test !any(isnan.(α))
    @test !any(isnan.(β))
    @test !any(isnan.(w))
    @test !any(isnan.(b))

    func_sq = x -> x .^ 2
    y_sq = func_sq.(x)
    R2_sq, α_sq, β_sq, w_sq, b_sq = FitSymbolic.fit_symbolic(
        x,
        y_sq,
        func_sq,
        I,
        O;
        max_iters = 30
    )

    @test size(R2_sq) == (I, O)
    @test !any(isnan.(R2_sq))

    return @test true
end

function test_reg()
    Random.seed!(42)
    I, O = 5, 3
    f = init_function(I, O; sample_size = b_size)

    Random.seed!(42)
    ps, st_kan = Lux.setup(Random.GLOBAL_RNG, f)

    x = randn(Float32, I, b_size) |> pu

    sf = FitSymbolic.SymFitter(
        conf,
        symbolic_lib = test_symb_lib,
    )

    fit = first(sf(ps, st_kan, f))

    x = randn(Float32, 2, 2)
    y = last(fit["i=1,o=1"])(x)

    @test isa(fit, Dict)
    @test length(fit) == I * O
    return @test !any(isnan.(y))
end

function test_symbolic_forward()
    Random.seed!(42)
    I, O, N = 2, 2, 10

    fit_dict = Dict{String, Tuple{String, Float32, Function}}()
    for i in 1:I, o in 1:O
        fit_dict["i=$i,o=$o"] = ("x", 0.99f0, x -> x)
    end

    α = ones(Float32, I, O)
    β = zeros(Float32, I, O)
    w = ones(Float32, I, O)
    b = zeros(Float32, I, O)

    min_grid = zeros(Float32, I)
    max_grid = ones(Float32, I)
    basis_function::AbstractBasis = RBF_basis(I, O, 2, N)

    sf = init_symbolic_function(
        basis_function,
        I,
        O,
        min_grid,
        max_grid,
        fit_dict,
        α,
        β,
        w,
        b
    )

    ps = Lux.initialparameters(Random.GLOBAL_RNG, sf)
    st = Lux.initialstates(Random.GLOBAL_RNG, sf)

    x = randn(Float32, I, O, N)
    y = sf(x, ps, st)

    # With identity function and α=1, β=0, w=1, b=0, output should equal input
    @test size(y) == size(x)
    @test all(isapprox.(y, x, atol = 1.0f-5))

    α2 = 2.0f0 .* ones(Float32, I, O)
    β2 = 1.0f0 .* ones(Float32, I, O)
    w2 = 0.5f0 .* ones(Float32, I, O)
    b2 = 0.25f0 .* ones(Float32, I, O)

    min_grid = zeros(Float32, I)
    max_grid = ones(Float32, I)

    sf2 = init_symbolic_function(
        basis_function,
        I,
        O,
        min_grid,
        max_grid,
        fit_dict,
        α2,
        β2,
        w2,
        b2
    )

    ps2 = Lux.initialparameters(Random.GLOBAL_RNG, sf2)

    y2 = sf2(x, ps2, st)
    expected = w2 .* (α2 .* x .+ β2) .+ b2
    @test all(isapprox.(y2, expected, atol = 1.0f-5))
    return @test true
end

function test_get_formula()
    Random.seed!(42)
    I, O = 2, 2

    fit_dict = Dict(
        "i=1,o=1" => ("sin", 0.95f0, x -> sin.(x)),
        "i=1,o=2" => ("x^2", 0.9f0, x -> x .^ 2),
    )

    α = [2.0f0 1.0f0; 0.5f0 1.0f0]
    β = [0.0f0 0.5f0; 0.0f0 0.0f0]
    w = [1.0f0 2.0f0; 1.0f0 1.0f0]
    b = [0.0f0 0.1f0; 0.0f0 0.0f0]

    min_grid = zeros(Float32, I)
    max_grid = ones(Float32, I)
    basis_function::AbstractBasis = RBF_basis(I, O, 2, 10)

    sf = init_symbolic_function(
        basis_function,
        I,
        O,
        min_grid,
        max_grid,
        fit_dict,
        α,
        β,
        w,
        b
    )

    ps = Lux.initialparameters(Random.GLOBAL_RNG, sf)

    formula_11 = get_formula(sf, ps, 1, 1)
    @test occursin("sin", formula_11)
    @test occursin("2.0", formula_11)

    formula_12 = get_formula(sf, ps, 1, 2)
    @test occursin("x^2", formula_12)
    @test occursin("2.0", formula_12)
    @test occursin("0.1", formula_12)

    formula_21 = get_formula(sf, ps, 2, 1)
    @test isa(formula_21, String)
    @test length(formula_21) > 0

    return @test true
end

function test_print_formulas()
    Random.seed!(42)
    I, O = 2, 2

    fit_dict = Dict(
        "i=1,o=1" => ("sin", 0.95f0, x -> sin.(x)),
    )

    α = ones(Float32, I, O)
    β = zeros(Float32, I, O)
    w = ones(Float32, I, O)
    b = zeros(Float32, I, O)

    min_grid = zeros(Float32, I)
    max_grid = ones(Float32, I)
    basis_function::AbstractBasis = RBF_basis(I, O, 2, 10)

    sf = init_symbolic_function(
        basis_function,
        I,
        O,
        min_grid,
        max_grid,
        fit_dict,
        α,
        β,
        w,
        b
    )

    ps = Lux.initialparameters(Random.GLOBAL_RNG, sf)

    print_formulas(sf, ps)
    return @test true
end

function test_symbolic_transfer()
    Random.seed!(42)
    rng = Random.MersenneTwister(1)
    optimizer = create_opt(conf)

    dataset = randn(rng, Float32, 32, 32, 1, 500)
    model = init_KAEM(dataset, conf, (32, 32, 1))
    x_test = first(model.train_loader) |> pu
    model, opt_state, ps, st_kan, st_lux, st_rng = prep_model(model, x_test, optimizer; rng = rng, MLIR = false)
    st = SymbolicTransfer(
        conf,
        model.lkhood.SEQ,
        model.lkhood.CNN;
        symbolic_lib_prior = test_symb_lib,
        symbolic_lib_llhood = test_symb_lib
    )
    @test st.transfer_ebm == true
    @test st.transfer_gen == true
    @test st.sym_fitter_ebm !== nothing
    @test st.sym_fitter_gen !== nothing

    model_sym = st(model, ps, st_kan; rng = rng)
    @test model_sym !== nothing
    return @test true
end

@testset "Symbolic Regression Tests" begin
    test_symbolic_functions()
    test_ols_wb()
    test_fit_affine()
    test_fit_symbolic()
    test_reg()
end

@testset "Symbolic Function Layer Tests" begin
    test_symbolic_forward()
    test_get_formula()
    test_print_formulas()
    test_symbolic_transfer()
end
