module KAEM_model

export KAEM, init_KAEM, generate_new

using ConfParser, Random, Lux, Accessors, ComponentArrays, Statistics
using Flux: DataLoader
using MultivariateStats: PCA, transform, fit
using MLDataDevices: cpu_device

using ..Utils

include("kan/univariate_functions.jl")
using .UnivariateFunctions

include("symbolic/symbolic_func.jl")
using .UnivariateFunctions

include("ebm/inverse_transform.jl")
using .InverseTransformSampling

include("ebm/ref_priors.jl")
using .RefPriors

include("ebm/ebm_model.jl")
include("gen/gen_model.jl")
using .EBM_Model
using .GeneratorModel

include("ebm/log_prior_fcns.jl")
using .LogPriorFCNs

include("posterior_sampling/population_xchange.jl")
using .PopulationXchange

include("posterior_sampling/encoder.jl")
using .EncoderModel

include("symbolic/reg.jl")
using .Reg

struct KAEM{T <: Float32} <: Lux.AbstractLuxLayer
    prior::EbmModel
    lkhood::GenModel
    encoder::EncoderWrapper
    train_loader::DataLoader
    test_loader::DataLoader
    batch_size::Int
    verbose::Bool
    N_t::Int
    sample_prior::Function
    posterior_sampler::Any
    xchange_func::Any
    train_step::Any
    ε::T
    file_loc::AbstractString
    MALA::Bool
    conf::ConfParse
    log_prior::AbstractLogPrior
    use_pca::Bool
    PCA_model::Any
    original_data_size::Tuple
    kan_regularizer::Any
    variational::Bool
end

function init_KAEM(
        dataset::AbstractArray{Float32},
        conf::ConfParse,
        x_shape::Tuple;
        file_loc::AbstractString = "logs/",
        rng::AbstractRNG = Random.default_rng(),
    )::KAEM{Float32}

    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))
    N_train = parse(Int, retrieve(conf, "TRAINING", "N_train"))
    N_test = parse(Int, retrieve(conf, "TRAINING", "N_test"))
    verbose = parse(Bool, retrieve(conf, "TRAINING", "verbose"))
    eps = parse(Float32, retrieve(conf, "TRAINING", "eps"))
    cnn = parse(Bool, retrieve(conf, "CNN", "use_cnn_lkhood"))
    seq = parse(Int, retrieve(conf, "SEQ", "sequence_length")) > 1

    train_data = seq ? dataset[:, :, 1:N_train] : dataset[:, :, :, 1:N_train]
    test_data =
        seq ? dataset[:, :, (N_train + 1):(N_train + N_test)] :
        dataset[:, :, :, (N_train + 1):(N_train + N_test)]

    original_data_size = x_shape
    use_pca = parse(Bool, retrieve(conf, "PCA", "use_pca"))
    pca_components = parse(Int, retrieve(conf, "PCA", "pca_components"))

    M = nothing
    if !cnn && !seq && use_pca
        train_data = reshape(train_data, :, size(train_data)[end])
        test_data = reshape(test_data, :, size(test_data)[end])
        M = fit(PCA, train_data; maxoutdim = pca_components)

        train_data = transform(M, train_data)
        test_data = transform(M, test_data)
        x_shape = (size(train_data, 1),)

        println("PCA model: num components = $pca_components")
    end


    train_loader = DataLoader(
        train_data .|> Float32,
        batchsize = batch_size,
        shuffle = true,
        rng = rng,
    )
    test_loader = DataLoader(test_data, batchsize = batch_size, shuffle = false)
    out_dim = (
        cnn ? size(dataset, 3) :
            (seq ? size(dataset, 1) : size(dataset, 1) * size(dataset, 2))
    )

    lkhood_fcn = retrieve(conf, "GeneratorModel", "spline_function")
    if lkhood_fcn == "FFT" || lkhood_fcn == "Cheby" || cnn
        update_llhood_grid = false
    end

    prior_model = init_EbmModel(conf; rng = rng)
    lkhood_model = init_GenModel(conf, x_shape; rng = rng)

    variational = parse(Bool, retrieve(conf, "VARIATIONAL", "use_variational"))
    encoder_model = init_encoder(conf, x_shape; rng = rng)

    η_init = parse(Float32, retrieve(conf, "POST_LANGEVIN", "initial_step_size"))
    N_t = parse(Int, retrieve(conf, "THERMODYNAMIC_INTEGRATION", "num_temps"))
    num_steps = parse(Int, retrieve(conf, "POST_LANGEVIN", "iters"))
    MALA = parse(Bool, retrieve(conf, "POST_LANGEVIN", "use_langevin"))

    N_t = max(N_t, 1)

    sample_prior =
        (m, n, p, sk, sl, r) ->
    sample_univariate(m.prior, n, p.ebm, sk.ebm, sl.ebm, sk.quad, r)

    verbose && println("Using $(Threads.nthreads()) threads.")

    return KAEM(
        prior_model,
        lkhood_model,
        encoder_model,
        train_loader,
        test_loader,
        batch_size,
        verbose,
        N_t,
        sample_prior,
        nothing,
        NoExchange(),
        nothing,
        eps,
        file_loc,
        MALA,
        conf,
        LogPriorUnivariate(eps, !prior_model.bool_config.contrastive_div),
        use_pca,
        M,
        original_data_size,
        Regularizer(conf, lkhood_model.CNN, lkhood_model.SEQ),
        variational,
    )
end

function init_from_file(file_loc::AbstractString, ckpt::Int)
    """Load a model from a checkpoint file."""
    saved_data = load(file_loc * "ckpt_epoch_$ckpt.jld2")
    model = saved_data["model"] |> deepcopy
    ps = convert(ComponentArray, saved_data["params"])
    st_kan = convert(NamedTuple, saved_data["kan_state"])
    st_lux = convert(NamedTuple, saved_data["lux_state"])
    return model, ps, st_kan, st_lux
end

function Lux.initialparameters(
        rng::AbstractRNG,
        model::KAEM{T},
    )::ComponentArray where {T <: Float32}
    return ComponentArray(
        ebm = Lux.initialparameters(rng, model.prior),
        gen = Lux.initialparameters(rng, model.lkhood),
        enc = Lux.initialparameters(rng, model.encoder),
    )
end

function Lux.initialstates(
        rng::AbstractRNG,
        model::KAEM{T},
    )::Tuple{NamedTuple, NamedTuple} where {T <: Float32}

    ebm_kan, ebm_lux = Lux.initialstates(rng, model.prior)
    gen_kan, gen_lux = Lux.initialstates(rng, model.lkhood)
    n, w = get_gausslegendre(model.prior, ebm_kan)
    st_quad = (nodes = n, weights = w)
    enc_lux = Lux.initialstates(rng, model.encoder)

    return (
        (ebm = ebm_kan, gen = gen_kan, quad = st_quad),
        (ebm = ebm_lux, gen = gen_lux, enc = enc_lux),
    )
end

function (model::KAEM{T})(
        ps,
        st_kan,
        st_lux,
        st_rng,
    ) where {T <: Float32}
    """
    Inference pass to generate a batch of data from the model.
    This is the same for both the standard and thermodynamic models.

    Args:
        model: The model.
        ps: The parameters of the model.
        st_kan: The states of the KAN model.
        st_lux: The states of the Lux model.
        num_samples: The number of samples to generate.
        rng: The random number generator.

    Returns:
        The generated data.
        Lux states of the prior.
        Lux states of the likelihood.
    """
    z, st_ebm = model.sample_prior(model, ps, st_kan, st_lux, st_rng)
    x̂, st_gen = model.lkhood.generator(ps.gen, st_kan.gen, st_lux.gen, z)
    return model.lkhood.output_activation(x̂), st_ebm, st_gen
end

end
