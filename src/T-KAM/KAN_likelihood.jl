module KAN_likelihood

export KAN_lkhood, init_KAN_lkhood, generate_from_z, importance_resampler, log_likelihood

using CUDA, KernelAbstractions, Tullio
using ConfParser, Random, Lux, LuxCUDA, Statistics, LinearAlgebra, ComponentArrays, Accessors
using NNlib: sigmoid_fast, tanh_fast, relu
using ChainRules: @ignore_derivatives
using Zygote: Buffer

include("univariate_functions.jl")
include("mixture_prior.jl")
include("resamplers.jl")
include("../utils.jl")
using .univariate_functions
using .Utils: device, next_rng, half_quant, full_quant, hq, fq
using .ebm_mix_prior
using .WeightResamplers

output_activation_mapping = Dict(
    "tanh" => tanh_fast,
    "sigmoid" => sigmoid_fast,
    "none" => identity
)

ll_models_flat = Dict(
    "l2" => (x::AbstractArray{full_quant}, x̂::AbstractArray{full_quant}; ε=eps(full_quant)) -> -dropdims( sum( @tullio(out[b, s, o] := (x[o, b] - x̂[o, s, b])^2) ; dims=3 ); dims=3 ), 
    "bernoulli" => (x::AbstractArray{full_quant}, x̂::AbstractArray{full_quant}; ε=eps(full_quant)) -> dropdims( sum( @tullio(out[b, s, o] := x[o, b] * log(x̂[s, o, b] + ε) + (1 - x[o, b]) * log(1 - x̂[s, o, b] + ε)) ; dims=3 ); dims=3 ),
)

lkhoohidden_dim_rgb = (x::AbstractArray{full_quant}, x̂::AbstractArray{full_quant}; ε=eps(full_quant)) -> -dropdims( sum( @tullio(out[b, s, h, w, c] := (x[h, w, c, b] - x̂[h, w, c, s, b])^2) ; dims=(3,4,5) ); dims=(3,4,5) )

function lkhoohidden_dim_seq(x::AbstractArray{full_quant}, x̂::AbstractArray{full_quant}; ε=eps(full_quant))
    log_x̂ = log.(x̂ .+ ε)    
    @tullio ll[b,s] := log_x̂[v,t,s,b] * x[v,t,b] # One-hot mask
    return ll ./ size(x̂, 1)
end

resampler_map = Dict(
    "residual" => residual_resampler,
    "systematic" => systematic_resampler,
    "stratified" => stratified_resampler,
)

struct KAN_lkhood <: Lux.AbstractLuxLayer
    Φ_fcns
    layernorm::Bool
    depth::Int
    out_size::Int
    σ_ε::half_quant
    σ_llhood::full_quant
    log_lkhoohidden_dim::Function
    output_activation::Function
    resample_z::Function
    generate_from_z::Function
    CNN::Bool
    seq_length::Int
end

function KAN_gen(
    lkhood, 
    ps, 
    st, 
    z::AbstractArray{half_quant}    
    )
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
    # KAN functions
    for i in 1:lkhood.depth
        z = fwd(lkhood.Φ_fcns[Symbol("$i")], ps[Symbol("$i")], st[Symbol("$i")], z)
        z = dropdims(sum(z, dims=1); dims=1)

        if lkhood.layernorm && i < lkhood.depth
            z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("ln_$i")], z, ps[Symbol("ln_$i")], st[Symbol("ln_$i")])
            @ignore_derivatives @reset st[Symbol("ln_$i")] = st_new
        end
    end

    return z, st
end

# CNN generator
function CNN_gen(
    lkhood, 
    ps, 
    st, 
    z::AbstractArray{half_quant}    
    )
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
    z = reshape(z, 1, 1, size(z)...)

    for i in 1:lkhood.depth
        z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("$i")], z, ps[Symbol("$i")], st[Symbol("$i")])
        @ignore_derivatives @reset st[Symbol("$i")] = st_new    

        z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("bn_$i")], z, ps[Symbol("bn_$i")], st[Symbol("bn_$i")])
        @ignore_derivatives @reset st[Symbol("bn_$i")] = st_new
    end

    z, st_new = Lux.apply(lkhood.Φ_fcns[Symbol("$(lkhood.depth+1)")], z, ps[Symbol("$(lkhood.depth+1)")], st[Symbol("$(lkhood.depth+1)")])
    @reset st[Symbol("$(lkhood.depth+1)")] = st_new

    return z, st
end

function scaled_dot_product_attention(
    Q::AbstractArray{half_quant}, 
    K::AbstractArray{half_quant}, 
    V::AbstractArray{half_quant}
)
    # Dot prod along hidden dim
    @tullio scores[j, t, b] := Q[i, j, b] * K[i, t, b]
    scores = scores ./ sqrt(size(Q, 1))

    # Causal mask
    mask = @ignore_derivatives tril(fill(-Inf, size(scores, 1), size(scores, 2))) .|> half_quant |> device 

    # Attention
    attn = softmax(scores .+ mask , dims=2)
    return @tullio out[i, j, b] := attn[j, t, b] * V[i, t, b]
end

# Seq generator
function SEQ_gen(lkhood, ps, st, z::AbstractArray{half_quant})
    
    z = fwd(lkhood.Φ_fcns[Symbol("1")], ps[Symbol("1")], st[Symbol("1")], z)
    z = dropdims(sum(z, dims=1); dims=1)

    z = reshape(z, size(z,1), 1, size(z,2))
    z = z .+ st[Symbol("pos_enc")]

    Q, st_q = Lux.apply(lkhood.Φ_fcns[Symbol("Query")], z, ps[Symbol("Query")], st[Symbol("Query")])
    K, st_k = Lux.apply(lkhood.Φ_fcns[Symbol("Key")], z, ps[Symbol("Key")], st[Symbol("Key")])
    V, st_v = Lux.apply(lkhood.Φ_fcns[Symbol("Value")], z, ps[Symbol("Value")], st[Symbol("Value")])

    z = z + scaled_dot_product_attention(Q, K, V)

    z, st_ln1 = Lux.apply(lkhood.Φ_fcns[Symbol("ln_1")], z, ps[Symbol("ln_1")], st[Symbol("ln_1")])
    z, st_ff = Lux.apply(lkhood.Φ_fcns[Symbol("2")], z, ps[Symbol("2")], st[Symbol("2")])
    z = z .+ st[Symbol("pos_enc")]  
    z, st_ln2 = Lux.apply(lkhood.Φ_fcns[Symbol("ln_2")], z, ps[Symbol("ln_2")], st[Symbol("ln_2")])
    z, st_out = Lux.apply(lkhood.Φ_fcns[Symbol("3")], z, ps[Symbol("3")], st[Symbol("3")])

    @ignore_derivatives begin
        @reset st[Symbol("Query")] = st_q
        @reset st[Symbol("Key")] = st_k
        @reset st[Symbol("Value")] = st_v
        @reset st[Symbol("ln_1")] = st_ln1
        @reset st[Symbol("2")] = st_ff
        @reset st[Symbol("ln_2")] = st_ln2
        @reset st[Symbol("3")] = st_out
    end

    return z, st
end 

function log_likelihood(
    lkhood, 
    ps, 
    st, 
    x::AbstractArray{full_quant}, 
    z::AbstractArray{half_quant};
    seed::Int=1,
    ε::full_quant=eps(full_quant),
    )
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
    Q, S, B = size(z)..., size(x)[end]

    x̂, st = lkhood.generate_from_z(lkhood, ps, st, z)

    # Add noise
    seed, rng = next_rng(seed)
    noise = lkhood.σ_ε * randn(rng, half_quant, size(x̂)..., B) |> device
    x̂ = lkhood.output_activation(x̂ .+ noise) |> fq # Accumulate across samples in full precision
    ll = lkhood.log_lkhoohidden_dim(x, x̂; ε=ε) ./ (2*lkhood.σ_llhood^2) 
    
    return ll, st, seed
end

function importance_resampler(
    weights::AbstractArray{full_quant};
    seed::Int=1,
    ESS_threshold::full_quant=full_quant(0.5),
    resampler::Function=systematic_sampler,
    verbose::Bool=false,
)
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
    output_dim::Int;
    lkhood_seed::Int=1,
    )

    q_size = (
        try 
            last(parse.(Int, retrieve(conf, "MIX_PRIOR", "layer_widths")))
        catch
            last(parse.(Int, split(retrieve(conf, "MIX_PRIOR", "layer_widths"), ",")))
        end
    )

    widths = (
        try 
            parse.(Int, retrieve(conf, "KAN_LIKELIHOOD", "widths"))
        catch
            parse.(Int, split(retrieve(conf, "KAN_LIKELIHOOD", "widths"), ","))
        end
    )

    CNN = parse(Bool, retrieve(conf, "CNN", "use_cnn_lkhood"))
    sequence_length = parse(Int, retrieve(conf, "SEQ", "sequence_length"))

    widths = (widths..., output_dim)
    first(widths) !== q_size && (error("First expert Φ_hidden_widths must be equal to the hidden dimension of the prior."))

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
    noise_var = parse(half_quant, retrieve(conf, "KAN_LIKELIHOOD", "generator_noise_var"))
    gen_var = parse(full_quant, retrieve(conf, "KAN_LIKELIHOOD", "generator_variance"))
    ESS_threshold = parse(full_quant, retrieve(conf, "TRAINING", "resampling_threshold_factor"))
    output_act = retrieve(conf, "KAN_LIKELIHOOD", "output_activation")
    resampler = retrieve(conf, "KAN_LIKELIHOOD", "resampler")
    verbose = parse(Bool, retrieve(conf, "TRAINING", "verbose"))
    resampler = resampler_map[resampler]

    resample_fcn = (weights, seed) -> @ignore_derivatives importance_resampler(weights; seed=seed, ESS_threshold=ESS_threshold, resampler=resampler, verbose=verbose)
    lkhoohidden_dim = retrieve(conf, "KAN_LIKELIHOOD", "likelihood_model")
    ll_model = ll_models_flat[lkhoohidden_dim]
    generate_fcn = KAN_gen

    output_activation = sequence_length > 1 ? (x -> softmax(x, dims=1)) : output_activation_mapping[output_act]

    Φ_functions = NamedTuple() 
    depth = length(widths)-1

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
        generate_fcn = CNN_gen
        ll_model = lkhoohidden_dim_rgb
        layernorm = false

        length(strides) != length(hidden_c) && (error("Number of strides must be equal to the number of hidden layers + 1."))
        length(k_size) != length(hidden_c) && (error("Number of kernel sizes must be equal to the number of hidden layers + 1."))
        length(paddings) != length(hidden_c) && (error("Number of paddings must be equal to the number of hidden layers + 1."))

        for i in eachindex(hidden_c[1:end-1])
            @reset Φ_functions[Symbol("$i")] = Lux.ConvTranspose((k_size[i], k_size[i]), hidden_c[i] => hidden_c[i+1], identity; stride=strides[i], pad=paddings[i])
            @reset Φ_functions[Symbol("bn_$i")] = Lux.BatchNorm(hidden_c[i+1], relu)
        end
        @reset Φ_functions[Symbol("$(length(hidden_c))")] = Lux.ConvTranspose((k_size[end], k_size[end]), hidden_c[end] => output_dim, identity; stride=strides[end], pad=paddings[end])

    elseif sequence_length > 1
        act = activation_mapping[retrieve(conf, "SEQ", "activation")]
        hidden_dim = parse(Int, retrieve(conf, "SEQ", "hidden_dim"))
        generate_fcn = SEQ_gen
        ll_model = lkhoohidden_dim_seq
        layernorm = true

        # Projection layer
        lkhood_seed, rng = next_rng(lkhood_seed)
        base_scale = (μ_scale * (full_quant(1) / √(full_quant(q_size)))
        .+ σ_base .* (randn(rng, full_quant, q_size, hidden_dim) .* full_quant(2) .- full_quant(1)) .* (full_quant(1) / √(full_quant(q_size))))
        
        @reset Φ_functions[Symbol("1")] = initialize_function(q_size, hidden_dim, base_scale)
        @reset Φ_functions[Symbol("Query")] = Lux.Dense(hidden_dim => hidden_dim, act)
        @reset Φ_functions[Symbol("Key")] = Lux.Dense(hidden_dim => hidden_dim, act)
        @reset Φ_functions[Symbol("Value")] = Lux.Dense(hidden_dim => hidden_dim, act)        
        @reset Φ_functions[Symbol("2")] = Lux.Dense(hidden_dim => hidden_dim, act)
        @reset Φ_functions[Symbol("ln_1")] = Lux.LayerNorm((hidden_dim, sequence_length); dims=1)
        @reset Φ_functions[Symbol("ln_2")] = Lux.LayerNorm((hidden_dim, sequence_length); dims=1)
        @reset Φ_functions[Symbol("3")] = Lux.Dense(hidden_dim => output_dim, identity)

        depth = 3
    else
        for i in eachindex(widths[1:end-1])
            lkhood_seed, rng = next_rng(lkhood_seed)
            base_scale = (μ_scale * (full_quant(1) / √(full_quant(widths[i])))
            .+ σ_base .* (randn(rng, full_quant, widths[i], widths[i+1]) .* full_quant(2) .- full_quant(1)) .* (full_quant(1) / √(full_quant(widths[i]))))
            @reset Φ_functions[Symbol("$i")] = initialize_function(widths[i], widths[i+1], base_scale)

            if (layernorm && i < depth)
                @reset Φ_functions[Symbol("ln_$i")] = Lux.LayerNorm(widths[i+1])
            end
        end
    end

    return KAN_lkhood(Φ_functions, layernorm, depth, output_dim, noise_var, gen_var, ll_model, output_activation, resample_fcn, generate_fcn, CNN, sequence_length)
end

function Lux.initialparameters(rng::AbstractRNG, lkhood::KAN_lkhood)

    ps = NamedTuple(Symbol("$i") => Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("$i")]) for i in 1:lkhood.depth)

    if lkhood.CNN
        @reset ps[Symbol("$(lkhood.depth+1)")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("$(lkhood.depth+1)")])
        for i in 1:lkhood.depth
            @reset ps[Symbol("bn_$i")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("bn_$i")])
        end
    elseif lkhood.seq_length > 1
        @reset ps[Symbol("Query")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("Query")])
        @reset ps[Symbol("Key")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("Key")])
        @reset ps[Symbol("Value")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("Value")])
    end

    if lkhood.layernorm 
        for i in 1:lkhood.depth-1
            @reset ps[Symbol("ln_$i")] = Lux.initialparameters(rng, lkhood.Φ_fcns[Symbol("ln_$i")]) 
        end
    end

    return ps 
end

function PositionEncoding(lkhood)
    hidden_dim = lkhood.Φ_fcns[Symbol("1")].out_dim
    max_len = lkhood.seq_length

    pe_vector = zeros(Float16, hidden_dim, max_len)
    position = range(1, max_len)
    div_term = exp.(-log(half_quant(10000)) .* range(1, hidden_dim, step=2) ./ hidden_dim)
    div_term = reshape(div_term, 1, floor(Int, hidden_dim/2))
    pe_vector[1:2:end, :] = transpose(sin.(position .* div_term))
    pe_vector[2:2:end, :] = transpose(cos.(position .* div_term))
    return pe_vector .|> half_quant
end

function Lux.initialstates(rng::AbstractRNG, lkhood::KAN_lkhood)

    st = NamedTuple(Symbol("$i") => Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("$i")]) |> hq for i in 1:lkhood.depth)

    if lkhood.CNN
        @reset st[Symbol("$(lkhood.depth+1)")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("$(lkhood.depth+1)")]) |> hq
        for i in 1:lkhood.depth
            @reset st[Symbol("bn_$i")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("bn_$i")]) |> hq
        end
    elseif lkhood.seq_length > 1
        @reset st[Symbol("Query")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("Query")]) |> hq
        @reset st[Symbol("Key")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("Key")]) |> hq
        @reset st[Symbol("Value")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("Value")]) |> hq
        @reset st[Symbol("pos_enc")] = PositionEncoding(lkhood) 
    end

    if lkhood.layernorm
        for i in 1:lkhood.depth-1
            @reset st[Symbol("ln_$i")] = Lux.initialstates(rng, lkhood.Φ_fcns[Symbol("ln_$i")]) |> hq
        end 
    end

    return st 
end

end