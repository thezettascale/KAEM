module EBM_Model

export EbmModel, init_EbmModel, get_gausslegendre

using ConfParser,
    Random,
    Distributions,
    Lux,
    Accessors,
    Statistics,
    LinearAlgebra,
    ComponentArrays,
    FastGaussQuadrature

using ..Utils
using ..UnivariateFunctions
using ..SymbolicFunctions
using ..RefPriors

include("quadrature.jl")
using .Quadrature

struct BoolConfig <: AbstractBoolConfig
    layernorm::Bool
    contrastive_div::Bool
    ula::Bool
    mixture_model::Bool
    use_attention_kernel::Bool
    train_props::Bool
    no_grid::Bool
end

struct EbmModel{T <: Float32} <: Lux.AbstractLuxLayer
    fcns_qp::Tuple{Vararg{Any}}
    layernorms::Tuple{Vararg{Lux.LayerNorm}}
    bool_config::BoolConfig
    depth::Int
    prior_type::AbstractString
    π_pdf::AbstractPrior
    p_size::Int
    q_size::Int
    s_size::Int
    quad::AbstractQuadrature
    N_quad::Int
    λ::T
    init_nodes
    init_weights
end

function init_EbmModel(conf::ConfParse; rng::AbstractRNG = Random.default_rng())
    widths = parse_config_array(Int, retrieve(conf, "EbmModel", "layer_widths"))
    spline_degree = parse(Int, retrieve(conf, "EbmModel", "spline_degree"))
    layernorm_bool = parse(Bool, retrieve(conf, "EbmModel", "layernorm"))
    base_activation = retrieve(conf, "EbmModel", "base_activation")
    spline_function = retrieve(conf, "EbmModel", "spline_function")
    grid_size = parse(Int, retrieve(conf, "EbmModel", "grid_size"))
    grid_update_ratio = parse(Float32, retrieve(conf, "EbmModel", "grid_update_ratio"))
    ε_scale = parse(Float32, retrieve(conf, "EbmModel", "ε_scale"))
    μ_scale = parse(Float32, retrieve(conf, "EbmModel", "μ_scale"))
    σ_base = parse(Float32, retrieve(conf, "EbmModel", "σ_base"))
    σ_spline = parse(Float32, retrieve(conf, "EbmModel", "σ_spline"))
    init_τ = parse(Float32, retrieve(conf, "EbmModel", "init_τ"))
    τ_trainable = parse(Bool, retrieve(conf, "EbmModel", "τ_trainable"))
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
    τ_trainable = spline_function == "B-spline" ? false : τ_trainable
    reg = parse(Float32, retrieve(conf, "MixtureModel", "λ_reg"))
    sample_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    P, Q = first(widths), last(widths)

    grid_range = parse.(Float32, retrieve(conf, "EbmModel", "grid_range"))
    prior_type = retrieve(conf, "EbmModel", "π_0")
    mixture_model = parse(Bool, retrieve(conf, "MixtureModel", "use_mixture_prior"))
    widths = mixture_model ? reverse(widths) : widths

    prior_domain = Dict(
        "ebm" => grid_range,
        "learnable_gaussian" => grid_range,
        "lognormal" => [0.0f0, 4.0f0],
        "gaussian" => [-1.5f0, 1.5f0],
        "uniform" => [0.0f0, 1.0f0],
    )[prior_type]

    eps = parse(Float32, retrieve(conf, "TRAINING", "eps"))
    ula = length(widths) > 2

    functions = []
    layernorms = Vector{Lux.LayerNorm}(undef, 0)

    outer_dim = mixture_model ? Q * sample_size : P * sample_size
    s_size = sample_size

    for i in eachindex(widths[1:(end - 1)])
        base_scale = (
            μ_scale * (1.0f0 / √(Float32(widths[i]))) .+
                σ_base .* (
                randn(rng, Float32, widths[i], widths[i + 1]) .* 2.0f0 .-
                    1.0f0
            ) .* (1.0f0 / √(Float32(widths[i])))
        )

        grid_range_i = i == 1 ? prior_domain : grid_range

        func = init_function(
            widths[i],
            widths[i + 1];
            spline_degree = spline_degree,
            base_activation = base_activation,
            spline_function = spline_function,
            grid_size = grid_size,
            grid_update_ratio = grid_update_ratio,
            grid_range = Tuple(grid_range_i),
            ε_scale = ε_scale,
            σ_base = base_scale,
            σ_spline = σ_spline,
            init_τ = init_τ,
            τ_trainable = τ_trainable,
            ε_ridge = eps,
            sample_size = s_size,
        )

        s_size = (i == 1 && !ula) ? outer_dim : s_size

        push!(functions, func)

        if layernorm_bool && i != 1
            push!(layernorms, Lux.LayerNorm(widths[i]))
        end
    end

    contrastive_div =
        parse(Bool, retrieve(conf, "TRAINING", "contrastive_divergence_training")) && !ula

    quad_fcn = GaussLegendreQuadrature()
    N_quad = parse(Int, retrieve(conf, "EbmModel", "GaussQuad_nodes"))

    ref_initializer = get(prior_map, prior_type, prior_map["uniform"])
    use_attention_kernel =
        parse(Bool, retrieve(conf, "MixtureModel", "use_attention_kernel"))
    train_props = parse(Bool, retrieve(conf, "MixtureModel", "train_proportions"))

    A = length(functions) > 0 ? typeof(functions[1].base_activation) : AbstractActivation

    init_nodes, init_weights = gausslegendre(N_quad)
    no_grid = spline_function == "FFT" || spline_function == "Cheby"

    return EbmModel{Float32}(
        Tuple(functions),
        Tuple(layernorms),
        BoolConfig(
            layernorm_bool,
            contrastive_div,
            ula,
            mixture_model,
            use_attention_kernel,
            train_props,
            no_grid
        ),
        length(widths) - 1,
        prior_type,
        ref_initializer(eps),
        P,
        Q,
        sample_size,
        quad_fcn,
        N_quad,
        reg,
        Float32.(init_nodes),
        Float32.(init_weights)
    )
end

function (ebm::EbmModel)(
        ps,
        st_kan,
        st_lyrnorm,
        z,
    )
    """
    Forward pass through the ebm-prior, returning the energy function.

    Args:
        ebm: The ebm-prior.
        ps: The parameters of the ebm-prior.
        st: The states of the ebm-prior.
        z: The component-wise latent samples to evaulate the measure on, (q, num_samples) or (p, num_samples)

    Returns:
        f: The energy function, (num_samples,) or (q, p, num_samples)
        st: The updated states of the ebm-prior.
    """

    mid_size = ebm.bool_config.mixture_model ? ebm.p_size : ebm.q_size
    outer_dim = ebm.bool_config.mixture_model ? ebm.q_size * ebm.s_size : ebm.p_size * ebm.s_size

    # Process first without layernorm
    z = Lux.apply(
        ebm.fcns_qp[1],
        z,
        ps.fcn[symbol_map[1]],
        st_kan[symbol_map[1]]
    )

    state = (2, st_lyrnorm, z .* 1.0f0)
    while first(state) <= ebm.depth
        i, st_lyrnorm_new, z_acc = state

        if ebm.bool_config.layernorm
            z_acc, st_layer_new = Lux.apply(
                ebm.layernorms[i - 1],
                z_acc,
                ps.layernorm[symbol_map[i]],
                st_lyrnorm_new[symbol_map[i]],
            )

            @reset st_lyrnorm_new[symbol_map[i]] = st_layer_new
        end

        z_acc = Lux.apply(ebm.fcns_qp[i], z_acc, ps.fcn[symbol_map[i]], st_kan[symbol_map[i]])

        if ebm.bool_config.layernorm
            z_acc = reshape(z_acc, mid_size, outer_dim)
        else
            z_acc = dropdims(sum(z_acc, dims = 1); dims = 1)
        end

        state = (i + 1, st_lyrnorm_new, z_acc)
    end

    _, st_lyrnorm_new, z = state
    z = (
        ebm.bool_config.ula ?
            z :
            reshape(z, ebm.q_size, ebm.p_size, ebm.s_size) .* 1.0f0
    )
    return z, st_lyrnorm_new
end

function Lux.initialparameters(
        rng::AbstractRNG,
        prior::EbmModel{T},
    )::NamedTuple where {T <: Float32}
    fcn_ps = NamedTuple(
        symbol_map[i] => Lux.initialparameters(rng, prior.fcns_qp[i]) for i in 1:prior.depth
    )
    layernorm_ps = init_optional_params(rng, prior.layernorms, prior.bool_config.layernorm)

    prior_ps = (
        π_μ = prior.prior_type == "learnable_gaussian" ?
            zeros(T, 1, prior.p_size) : [0.0f0],
        π_σ = prior.prior_type == "learnable_gaussian" ?
            ones(T, 1, prior.p_size) : [0.0f0],
        α = !prior.bool_config.mixture_model ? [0.0f0] :
            (
                !prior.bool_config.use_attention_kernel ?
                glorot_uniform(rng, Float32, prior.q_size, prior.p_size) : [0.0f0]
            ),
    )

    if !prior.bool_config.train_props && !prior.bool_config.use_attention_kernel
        @reset prior_ps.α = (prior_ps.α .* 0 .+ 1) ./ prior.p_size
    end


    attention_ps = (
        Q = prior.bool_config.use_attention_kernel ?
            glorot_normal(rng, Float32, prior.q_size, prior.p_size) : [0.0f0],
        K = prior.bool_config.use_attention_kernel ?
            glorot_normal(rng, Float32, prior.q_size, prior.p_size) : [0.0f0],
    )

    return (
        fcn = fcn_ps,
        dist = prior_ps,
        layernorm = layernorm_ps,
        attention = attention_ps,
    )
end

function Lux.initialstates(
        rng::AbstractRNG,
        prior::EbmModel{T},
    )::Tuple{NamedTuple, NamedTuple} where {T <: Float32}
    fcn_st = NamedTuple(
        symbol_map[i] => Lux.initialstates(rng, prior.fcns_qp[i]) for i in 1:prior.depth
    )
    st_lyrnorm = init_optional_states(rng, prior.layernorms, prior.bool_config.layernorm)
    return fcn_st, st_lyrnorm
end

end
