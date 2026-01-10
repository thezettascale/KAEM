export PangEBM, init_PangEBM, log_likelihood

struct PangEBM{T <: Float32} <: Lux.AbstractLuxLayer
    generator::PangGenerator
    energy_net::EnergyMLP
    latent_dim::Int
    x_shape::Tuple{Vararg{Int}}
    batch_size::Int
    prior_sgld_steps::Int
    prior_sgld_step_size::Float32
    post_sgld_steps::Int
    post_sgld_step_size::Float32
    noise_scale::Float32
end

function init_PangEBM(
        conf::ConfParse,
        x_shape::Tuple{Vararg{Int}};
        rng::AbstractRNG = Random.default_rng(),
    )
    latent_dim = parse(Int, retrieve(conf, "PANG", "latent_dim"))
    gen_channels = parse.(Int, retrieve(conf, "PANG", "generator_channels"))
    energy_widths = parse.(Int, retrieve(conf, "PANG", "energy_widths"))
    kernel_size = parse(Int, retrieve(conf, "PANG", "kernel_size"))
    batch_size = parse(Int, retrieve(conf, "TRAINING", "batch_size"))

    prior_sgld_steps = parse(Int, retrieve(conf, "PANG", "prior_sgld_steps"))
    prior_sgld_step_size = parse(Float32, retrieve(conf, "PANG", "prior_sgld_step_size"))
    post_sgld_steps = parse(Int, retrieve(conf, "PANG", "post_sgld_steps"))
    post_sgld_step_size = parse(Float32, retrieve(conf, "PANG", "post_sgld_step_size"))
    noise_scale = parse(Float32, retrieve(conf, "PANG", "noise_scale"))

    generator = init_pang_generator(x_shape, latent_dim, gen_channels, kernel_size)
    energy_net = init_energy_mlp(latent_dim, energy_widths)

    return PangEBM{Float32}(
        generator,
        energy_net,
        latent_dim,
        x_shape,
        batch_size,
        prior_sgld_steps,
        prior_sgld_step_size,
        post_sgld_steps,
        post_sgld_step_size,
        noise_scale,
    )
end

function Lux.initialparameters(rng::AbstractRNG, model::PangEBM)
    return (
        gen = Lux.initialparameters(rng, model.generator),
        ebm = Lux.initialparameters(rng, model.energy_net),
    )
end

function Lux.initialstates(rng::AbstractRNG, model::PangEBM)
    return (
        gen = Lux.initialstates(rng, model.generator),
        ebm = Lux.initialstates(rng, model.energy_net),
    )
end

function log_likelihood(model::PangEBM, x, z, ps, st; σ²::Float32 = 1.0f0)
    x_gen, st_gen = generate(model.generator, z, ps.gen, st.gen)
    ll = -sum((x .- x_gen) .^ 2, dims = (1, 2, 3)) ./ (2.0f0 * σ²)
    return dropdims(ll; dims = (1, 2, 3)), st_gen
end

function (model::PangEBM)(ps, st, z)
    x_gen, st_gen = generate(model.generator, z, ps.gen, st.gen)
    return x_gen, (gen = st_gen, ebm = st.ebm)
end
