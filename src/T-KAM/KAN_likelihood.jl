module KAN_likelihood

export KAN_lkhood, init_KAN_lkhood, generate_from_z, importance_resampler, log_likelihood

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays, Accessors
using NNlib: sigmoid_fast, tanh_fast, relu, gelu, sigmoid, tanh
using ChainRules: @ignore_derivatives
using Zygote: Buffer

include("univariate_functions.jl")
include("EBM_prior.jl")
include("resamplers.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng, half_quant, full_quant, hq, fq
using .ebm_ebm_prior
using .WeightResamplers

output_activation_mapping = Dict(
    "tanh" => tanh_fast, 
    "sigmoid" => sigmoid_fast,
    "none" => identity,
)

# Fcns for the Vanilla model
function lkhood_rgb(
    x::AbstractArray{T}, 
    x̂::AbstractArray{T}; 
    ε::T=eps(T)
    ) where {T<:half_quant}
    -dropdims( sum( (x .- permutedims(x̂, [1, 2, 3, 5, 4])).^ 2, dims=(1,2,3) ); dims=(1,2,3) )
end

function lkhood_seq(
    x::AbstractArray{T}, 
    x̂::AbstractArray{T}; 
    ε::T=eps(T)
    ) where {T<:half_quant}
    log_x̂ = log.(x̂ .+ ε)    
    ll = dropdims(sum(permutedims(log_x̂, [1, 2, 4, 3]) .* x, dims=(1,2)), dims=(1,2)) # One-hot encoded cross-entropy
    return ll ./ size(x̂, 1)
end
 
# Fcns for model with Lagenvin methods
function cross_entropy(
    x::AbstractArray{T}, 
    x̂::AbstractArray{T};
    t::Union{AbstractArray{T}, T}=device([one(half_quant)]), 
    ε::T=eps(half_quant),
    σ::T=one(half_quant),
    ) where {T<:half_quant}
    ll = log.(x̂ .+ ε) .* x ./ size(x, 1)
    return t .* dropdims(sum(ll; dims=(1,2)); dims=(1,2)) ./ (2*σ^2)
end

function l2(
    x::AbstractArray{T}, 
    x̂::AbstractArray{T}; 
    t::Union{AbstractArray{T}, T}=device([one(half_quant)]), 
    ε::T=eps(half_quant),
    σ::T=one(half_quant),
    ) where {T<:half_quant}
    ll = -(x - x̂).^2
    return t .* dropdims(sum(ll; dims=(1,2,3)); dims=(1,2,3)) ./ (2*σ^2)
end

resampler_map = Dict(
    "residual" => residual_resampler,
    "systematic" => systematic_resampler,
    "stratified" => stratified_resampler,
)

struct KAN_lkhood{T<:half_quant} <: Lux.AbstractLuxLayer
    Φ_fcns
    layernorm::Bool
    batchnorm::Bool
    depth::Int
    out_size::Int
    σ_llhood::T
    log_lkhood::Function
    output_activation::Function
    x_shape::Tuple{Vararg{Int}}
    resample_z::Function
    generate_from_z::Function
    CNN::Bool
    seq_length::Int
    d_model::Int
    MALA_ll_fcn::Function
end

function KAN_gen(
    lkhood, 
    ps, 
    st, 
    z::AbstractArray{T}    
    ) where {T<:half_quant}
    """
    Generate data from the KAN likelihood model.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        seed: The seed for the random number generator.

    Returns:
        The generated data.
        The updated seed.
    """
    num_samples = size(z)[end]
    z = dropdims(sum(z, dims=2), dims=2)

    # KAN functions
    for i in 1:lkhood.depth
        z = fwd(lkhood.Φ_fcns[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = dropdims(sum(z, dims=1); dims=1)

        if lkhood.layernorm && i < lkhood.depth
            z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("ln_$i")], z, ps[Symbol("ln_$i")], st[Symbol("ln_$i")])
            @reset st[Symbol("ln_$i")] = st_new
        end
    end

    return reshape(z, lkhood.x_shape..., num_samples), st
end

# CNN generator
function CNN_gen(
    lkhood, 
    ps, 
    st, 
    z::AbstractArray{T}    
    ) where {T<:half_quant}
    """
    Generate data from the CNN likelihood model.
    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        seed: The seed for the random number generator.
    Returns:
        The generated data.
        The updated seed.
    """
    z = reshape(sum(z, dims=2), 1, 1, first(size(z)), last(size(z)))

    for i in 1:lkhood.depth
        z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("$i")], z, ps[Symbol("$i")], st[Symbol("$i")])
        @reset st[Symbol("$i")] = st_new    

        if lkhood.batchnorm
            z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("bn_$i")], z, ps[Symbol("bn_$i")], st[Symbol("bn_$i")])
            @reset st[Symbol("bn_$i")] = st_new
        end
    end

    z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("$(lkhood.depth+1)")], z, ps[Symbol("$(lkhood.depth+1)")], st[Symbol("$(lkhood.depth+1)")])
    @reset st[Symbol("$(lkhood.depth+1)")] = st_new

    return z, st
end

function scaled_dot_product_attention(
    Q::AbstractArray{T},
    K::AbstractArray{T},
    V::AbstractArray{T},
    d_model::Int
    ) where {T<:half_quant}

    d_model = sqrt(d_model) |> T
    @tullio QK[t, i, b] := Q[d, t, b] * K[d, i, b] / d_model
    QK = softmax(QK, dims=2)
    return @tullio out[d, t, b] := QK[t, i, b] * V[d, i, b]
end

function SEQ_gen(
    lkhood, 
    ps, 
    st, 
    z::AbstractArray{T}
    ) where {T<:half_quant}
    """
    Generate data from the Transformer decoder.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        z: The latent variable.

    Returns:
        The generated data.
        The updated seed.
    """
    z = sum(z, dims=2)
    
    # Projection
    z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("1")], z, ps[Symbol("1")], st[Symbol("1")])
    @reset st[Symbol("1")] = st_new
    z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("ln_1")], z, ps[Symbol("ln_1")], st[Symbol("ln_1")])
    @reset st[Symbol("ln_1")] = st_new

    z_prev = z
    for t in 2:lkhood.seq_length

        # Self-attention
        Q, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("Q")], z, ps[Symbol("Q")], st[Symbol("Q")])
        @reset st[Symbol("Q")] = st_new
        K, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("K")], z, ps[Symbol("K")], st[Symbol("K")])
        @reset st[Symbol("K")] = st_new
        V, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("V")], z, ps[Symbol("V")], st[Symbol("V")])
        @reset st[Symbol("V")] = st_new

        attn = scaled_dot_product_attention(Q, K, V, lkhood.d_model)
        z = z .+ attn

        # Feed forward
        z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("2")], z, ps[Symbol("2")], st[Symbol("2")])
        @reset st[Symbol("2")] = st_new
        z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("ln_2")], z[:, end:end, :], ps[Symbol("ln_2")], st[Symbol("ln_2")])
        @reset st[Symbol("ln_2")] = st_new

        z = cat(z_prev, z, dims=2)
        z_prev = z
    end

    # Output layer
    z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("3")], z, ps[Symbol("3")], st[Symbol("3")])
    @reset st[Symbol("3")] = st_new

    return z, st
end

function log_likelihood(
    lkhood, 
    ps, 
    st, 
    x::AbstractArray{T}, 
    z::AbstractArray{T};
    seed::Int=1,
    ε::T=eps(T),
    ) where {T<:half_quant}
    """
    Evaluate the unnormalized log-likelihood of the KAN generator.

    Args:
        lkhood: The likelihood model.
        ps: The parameters of the likelihood model.
        st: The states of the likelihood model.
        x: The data.
        z: The latent variable.
        tempered: Whether to use tempered likelihood.
        seed: The seed for the random number generator.

    Returns:
        The unnormalized log-likelihood.
        The updated seed.
    """
    Q, P, S, B = size(z)..., size(x)[end]

    x̂, st = lkhood.generate_from_z(lkhood, ps, st, z)

    # Add noise
    seed, rng = next_rng(seed)
    noise = lkhood.σ_llhood * randn(rng, T, size(x̂)..., B) |> device
    x̂ = lkhood.output_activation(x̂ .+ noise) 
    ll = lkhood.log_lkhood(x, x̂; ε=ε) ./ (2*lkhood.σ_llhood^2) 
    
    return ll, st, seed
end

function importance_resampler(
    weights::AbstractArray{U};
    seed::Int=1,
    ESS_threshold::U=full_quant(0.5),
    resampler::Function=systematic_sampler,
    verbose::Bool=false,
) where {U<:full_quant}
    """
    Filter the latent variable for a index of the Steppingstone sum using residual resampling.

    Args:
        logllhood: A matrix of log-likelihood values.
        weights: The weights of the population.
        t_resample: The temperature at which the last resample occurred.
        t2: The temperature at which to update the weights.
        seed: Random seed for reproducibility.
        ESS_threshold: The threshold for the effective sample size.
        resampler: The resampling function.

    Returns:
        - The resampled indices.
        - The updated seed.
    """
    B, N = size(weights)

    # Check effective sample size
    ESS = dropdims(1 ./ sum(weights.^2, dims=2); dims=2)
    ESS_bool = ESS .< ESS_threshold*N
    
    # Only resample when needed 
    verbose && (any(ESS_bool) && println("Resampling!"))
    any(ESS_bool) && return resampler(cpu_device()(weights), cpu_device()(ESS_bool), B, N; seed=seed)
    return repeat(collect(1:N)', B, 1), seed
end

function init_KAN_lkhood(
    conf::ConfParse,
    x_shape::Tuple{Vararg{Int}};
    lkhood_seed::Int=1,
    )

    prior_widths = (
        try 
            parse.(Int, retrieve(conf, "EBM_PRIOR", "layer_widths"))
        catch
            parse.(Int, split(retrieve(conf, "EBM_PRIOR", "layer_widths"), ","))
        end
    )

    q_size = length(prior_widths) > 2 ? first(prior_widths) : last(prior_widths)

    widths = (
        try 
            parse.(Int, retrieve(conf, "KAN_LIKELIHOOD", "widths"))
        catch
            parse.(Int, split(retrieve(conf, "KAN_LIKELIHOOD", "widths"), ","))
        end
    )

    CNN = parse(Bool, retrieve(conf, "CNN", "use_cnn_lkhood"))
    sequence_length = parse(Int, retrieve(conf, "SEQ", "sequence_length"))

    output_dim = CNN ? last(x_shape) : (sequence_length > 1 ? first(x_shape) : prod(x_shape))

    widths = (widths..., output_dim)
    first(widths) !== q_size && (error("First expert Φ_hidden_widths must be equal to the hidden dimension of the prior.", widths, " != ", q_size))

    spline_degree = parse(Int, retrieve(conf, "KAN_LIKELIHOOD", "spline_degree"))
    layernorm = parse(Bool, retrieve(conf, "KAN_LIKELIHOOD", "layer_norm"))
    base_activation = retrieve(conf, "KAN_LIKELIHOOD", "base_activation")
    spline_function = retrieve(conf, "KAN_LIKELIHOOD", "spline_function")
    grid_size = parse(Int, retrieve(conf, "KAN_LIKELIHOOD", "grid_size"))
    grid_update_ratio = parse(half_quant, retrieve(conf, "KAN_LIKELIHOOD", "grid_update_ratio"))
    grid_range = parse.(half_quant, retrieve(conf, "KAN_LIKELIHOOD", "grid_range"))
    ε_scale = parse(half_quant, retrieve(conf, "KAN_LIKELIHOOD", "ε_scale"))
    μ_scale = parse(full_quant, retrieve(conf, "KAN_LIKELIHOOD", "μ_scale"))
    σ_base = parse(full_quant, retrieve(conf, "KAN_LIKELIHOOD", "σ_base"))
    σ_spline = parse(full_quant, retrieve(conf, "KAN_LIKELIHOOD", "σ_spline"))
    init_τ = parse(full_quant, retrieve(conf, "KAN_LIKELIHOOD", "init_τ"))
    τ_trainable = parse(Bool, retrieve(conf, "KAN_LIKELIHOOD", "τ_trainable"))
    τ_trainable = spline_function == "B-spline" ? false : τ_trainable
    gen_var = parse(half_quant, retrieve(conf, "KAN_LIKELIHOOD", "generator_variance"))
    ESS_threshold = parse(full_quant, retrieve(conf, "TRAINING", "resampling_threshold_factor"))
    output_act = retrieve(conf, "KAN_LIKELIHOOD", "output_activation")
    resampler = retrieve(conf, "KAN_LIKELIHOOD", "resampler")
    verbose = parse(Bool, retrieve(conf, "TRAINING", "verbose"))
    resampler = resampler_map[resampler]
    batchnorm = false

    resample_fcn = (weights, seed) -> @ignore_derivatives importance_resampler(weights; seed=seed, ESS_threshold=ESS_threshold, resampler=resampler, verbose=verbose)
    ll_model = lkhood_rgb
    generate_fcn = KAN_gen

    output_activation = sequence_length > 1 ? (x -> softmax(x, dims=1)) : get(output_activation_mapping, output_act, identity)
    sampling_fcn = sequence_length > 1 ? cross_entropy : l2

    println("Using KAN likelihood with ", sequence_length > 1 ? "cross-entropy" : "L2", " loss function.")

    Φ_functions = NamedTuple() 
    depth = length(widths)-1
    d_model = 0

    initialize_function = (in_dim, out_dim, base_scale) -> init_function(
            in_dim,
            out_dim;
            spline_degree=spline_degree,
            base_activation=base_activation,
            spline_function=spline_function,
            grid_size=grid_size,
            grid_update_ratio=grid_update_ratio,
            grid_range=Tuple(grid_range),
            ε_scale=ε_scale,
            σ_base=base_scale,
            σ_spline=σ_spline,
            init_τ=init_τ,
            τ_trainable=τ_trainable,
        )

    if CNN
        channels = parse.(Int, retrieve(conf, "CNN", "hidden_feature_dims"))
        hidden_c = (q_size, channels...)
        depth = length(hidden_c)-1
        strides = parse.(Int, retrieve(conf, "CNN", "strides"))
        k_size = parse.(Int, retrieve(conf, "CNN", "kernel_sizes"))
        paddings = parse.(Int, retrieve(conf, "CNN", "paddings"))
        act = activation_mapping[retrieve(conf, "CNN", "activation")]
        batchnorm = parse(Bool, retrieve(conf, "CNN", "batchnorm"))
        generate_fcn = CNN_gen
        layernorm = false

        length(strides) != length(hidden_c) && (error("Number of strides must be equal to the number of hidden layers + 1."))
        length(k_size) != length(hidden_c) && (error("Number of kernel sizes must be equal to the number of hidden layers + 1."))
        length(paddings) != length(hidden_c) && (error("Number of paddings must be equal to the number of hidden layers + 1."))

        for i in eachindex(hidden_c[1:end-1])
            @reset Φ_functions[Symbol("$i")] = Lux.ConvTranspose((k_size[i], k_size[i]), hidden_c[i] => hidden_c[i+1], identity; stride=strides[i], pad=paddings[i])
            if batchnorm
                @reset Φ_functions[Symbol("bn_$i")] = Lux.BatchNorm(hidden_c[i+1], act)
            end
        end
        @reset Φ_functions[Symbol("$(length(hidden_c))")] = Lux.ConvTranspose((k_size[end], k_size[end]), hidden_c[end] => output_dim, identity; stride=strides[end], pad=paddings[end])

    elseif sequence_length > 1
        
        act = gelu
        generate_fcn = SEQ_gen
        ll_model = lkhood_seq
        # Single block Transformer decoder
        d_model = parse(Int, retrieve(conf, "SEQ", "d_model"))

        # Projection
        @reset Φ_functions[Symbol("1")] = Lux.Dense(q_size => d_model)
        @reset Φ_functions[Symbol("ln_1")] = Lux.LayerNorm((d_model, 1), gelu)

        # Query, Key, Value
        @reset Φ_functions[Symbol("Q")] = Lux.Dense(d_model => d_model)
        @reset Φ_functions[Symbol("K")] = Lux.Dense(d_model => d_model) 
        @reset Φ_functions[Symbol("V")] = Lux.Dense(d_model => d_model)

        # Feed forward
        @reset Φ_functions[Symbol("2")] = Lux.Dense(d_model => d_model)
        @reset Φ_functions[Symbol("ln_2")] = Lux.LayerNorm((d_model, 1), gelu)

        # Output layer
        @reset Φ_functions[Symbol("3")] = Lux.Dense(d_model => output_dim)
        depth = 3
    else
        for i in eachindex(widths[1:end-1])
            lkhood_seed, rng = next_rng(lkhood_seed)
            base_scale = (μ_scale * (one(full_quant) / √(full_quant(widths[i])))
            .+ σ_base .* (randn(rng, full_quant, widths[i], widths[i+1]) .* full_quant(2) .- one(full_quant)) .* (one(full_quant) / √(full_quant(widths[i]))))
            @reset Φ_functions[Symbol("$i")] = initialize_function(widths[i], widths[i+1], base_scale)

            if (layernorm && i < depth)
                @reset Φ_functions[Symbol("ln_$i")] = Lux.LayerNorm(widths[i+1])
            end
        end
    end

    return KAN_lkhood(
        Φ_functions, 
        layernorm, 
        batchnorm, 
        depth, 
        output_dim, 
        gen_var, 
        ll_model, 
        output_activation, 
        x_shape, 
        resample_fcn, 
        generate_fcn, 
        CNN, 
        sequence_length, 
        d_model,
        sampling_fcn
        )
end

function Lux.initialparameters(rng::AbstractRNG, lkhood::KAN_lkhood)

    ps = NamedTuple(Symbol("$i") => Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("$i")]) for i in 1:lkhood.depth)

    if lkhood.CNN
        @reset ps[Symbol("$(lkhood.depth+1)")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("$(lkhood.depth+1)")])
        if lkhood.batchnorm
            for i in 1:lkhood.depth
                @reset ps[Symbol("bn_$i")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("bn_$i")])
            end
        end
    end

    if lkhood.layernorm 
        for i in 1:lkhood.depth-1
            @reset ps[Symbol("ln_$i")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("ln_$i")]) 
        end
    end

    if lkhood.seq_length > 1
        @reset ps[Symbol("Q")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("Q")])
        @reset ps[Symbol("K")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("K")])
        @reset ps[Symbol("V")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("V")])
    end

    return ps 
end

function Lux.initialstates(rng::AbstractRNG, lkhood::KAN_lkhood)

    st = NamedTuple(Symbol("$i") => Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("$i")]) |> hq for i in 1:lkhood.depth)

    if lkhood.CNN
        @reset st[Symbol("$(lkhood.depth+1)")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("$(lkhood.depth+1)")]) |> hq
        
        if lkhood.batchnorm
            for i in 1:lkhood.depth
                @reset st[Symbol("bn_$i")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("bn_$i")]) |> hq
            end
        end
    end

    if lkhood.layernorm
        for i in 1:lkhood.depth-1
            @reset st[Symbol("ln_$i")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("ln_$i")]) |> hq
        end 
    end

    if lkhood.seq_length > 1
        @reset st[Symbol("Q")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("Q")])
        @reset st[Symbol("K")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("K")])
        @reset st[Symbol("V")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("V")])
    end

    return st 
end

end