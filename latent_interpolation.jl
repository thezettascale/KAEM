using ConfParser, Random, JLD2, ComponentArrays, Lux, Reactant, Statistics, LinearAlgebra
using CairoMakie, LaTeXStrings, Colors

ENV["DEVICE"] = "gpu"
CairoMakie.activate!(type = "png")

dataset = length(ARGS) >= 1 ? ARGS[1] : "CELEBA"
mode = length(ARGS) >= 2 ? ARGS[2] : "vanilla"

dataset_configs = Dict(
    "CELEBA" => (config = "config/celeba_config.ini", resize = (64, 64)),
    "SVHN" => (config = "config/svhn_config.ini", resize = (32, 32)),
    "CIFAR10" => (config = "config/cifar_config.ini", resize = (32, 32)),
)

haskey(dataset_configs, dataset) || error("Unknown dataset: $dataset. Use one of: $(keys(dataset_configs))")
ds = dataset_configs[dataset]

file_loc = (
    mode == "thermo" ? "logs/Thermodynamic/$(dataset)/ULA/mixture/" :
        "logs/Vanilla/$(dataset)/ULA/mixture/"
)
save_dir = "figures/interpolations/$(dataset)/$(mode)/"
mkpath(save_dir)

num_interp_steps = 8
num_pairs = 6
num_prior_samples = 500
num_density_dims = 3

conf = ConfParse(ds.config)
parse_conf!(conf)
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
ENV["DEVICE"] = retrieve(conf, "TRAINING", "device")
ENV["PERCEPTUAL"] = retrieve(conf, "TRAINING", "use_perceptual_loss")

include("src/utils.jl")
using .Utils

include("src/pipeline/trainer.jl")
using .trainer
using .trainer: seed_rand
using .trainer.KAEM_model: load_params
saved_data = load(file_loc * "saved_model.jld2")
rng = Random.MersenneTwister(1)
t = init_trainer(rng, conf, dataset; img_resize = ds.resize, file_loc = file_loc, save_model = false)

t.ps = load_params(saved_data) |> pu
t.st_kan = saved_data["kan_state"] |> pu
t.st_lux = saved_data["lux_state"] |> pu

model = t.model
ps = t.ps
st_kan = t.st_kan
st_lux = Lux.testmode(t.st_lux)
q_size = model.prior.q_size
batch_size = model.batch_size

# Sample from prior
num_batches = num_prior_samples ÷ batch_size
z_all = zeros(Float32, q_size, 1, num_batches * batch_size)
for i in 1:num_batches
    st_rng = seed_rand(model; rng = rng)
    z, _ = Reactant.@jit model.sample_prior(ps, st_kan, st_lux, st_rng)
    idx = ((i - 1) * batch_size + 1):(i * batch_size)
    z_all[:, :, idx] = Array(z)
end
println("Sampled $(size(z_all, 3)) latent vectors")

# Compile decoder
function decode(ps_gen, st_kan_gen, st_lux_gen, z)
    x̂, _ = model.lkhood.generator(ps_gen, st_kan_gen, st_lux_gen, z)
    return model.lkhood.output_activation(x̂)
end

z_dummy = pu(zeros(Float32, q_size, 1, batch_size))
decode_compiled = Reactant.@compile decode(ps.gen, st_kan.gen, st_lux.gen, z_dummy)
println("Decoder compiled.")

# SLERP
function slerp(z1, z2, t)
    z1_flat = vec(z1)
    z2_flat = vec(z2)
    n1 = z1_flat / max(norm(z1_flat), eps(Float32))
    n2 = z2_flat / max(norm(z2_flat), eps(Float32))
    omega = acos(clamp(dot(n1, n2), -1.0f0, 1.0f0))
    if omega < 1.0f-6
        return (1.0f0 - t) .* z1 .+ t .* z2
    end
    s = sin(omega)
    return (sin((1.0f0 - t) * omega) / s) .* z1 .+ (sin(t * omega) / s) .* z2
end

# Pairs far apart
N = size(z_all, 3)
pair_dists = Dict{Tuple{Int, Int}, Float32}()
for i in 1:N, j in (i + 1):N
    pair_dists[(i, j)] = sum((z_all[:, :, i] .- z_all[:, :, j]) .^ 2)
end
sorted_pairs = sort(collect(pair_dists); by = last, rev = true)

pairs = Tuple{Int, Int}[]
used = Set{Int}()
for (p, _) in sorted_pairs
    (p[1] in used || p[2] in used) && continue
    push!(pairs, p)
    push!(used, p[1], p[2])
    length(pairs) >= num_pairs && break
end
println("Selected pairs: $pairs")

# Interpolate and plot each pair
num_cols = num_interp_steps + 2
ts = range(0.0f0, 1.0f0; length = num_cols)

for (pair_idx, (i, j)) in enumerate(pairs)
    z_a = z_all[:, :, i:i]
    z_b = z_all[:, :, j:j]

    z_batch = repeat(z_a, 1, 1, batch_size)
    for (ci, t_val) in enumerate(ts)
        z_batch[:, :, ci] .= slerp(z_a, z_b, t_val)
    end
    x_decoded = Array(decode_compiled(ps.gen, st_kan.gen, st_lux.gen, pu(z_batch)))

    all_rgb = Matrix{RGB{Float32}}[]
    for ci in 1:num_cols
        raw = clamp.(x_decoded[:, :, :, ci], 0.0f0, 1.0f0)
        rgb = RGB.(raw[:, :, 1], raw[:, :, 2], raw[:, :, 3])
        push!(all_rgb, rot180(rgb))
    end

    # Dims that are multimodal and have good z_A/z_B separation
    function count_modes(samples, nbins = 30)
        lo, hi = extrema(samples)
        edges = range(lo - 0.01f0, hi + 0.01f0; length = nbins + 1)
        counts = zeros(Int, nbins)
        for s in samples
            bin = clamp(searchsortedlast(edges, s), 1, nbins)
            counts[bin] += 1
        end

        # Smooth with a 3-wide moving average to suppress noise
        smoothed = [mean(counts[max(1, i - 1):min(nbins, i + 1)]) for i in 1:nbins]

        # Count local maxima
        n_modes = 0
        for i in 2:(nbins - 1)
            if smoothed[i] > smoothed[i - 1] && smoothed[i] > smoothed[i + 1]
                n_modes += 1
            end
        end
        return max(n_modes, 1)
    end

    zdiff = abs.(vec(z_a[:, 1, 1]) .- vec(z_b[:, 1, 1]))
    mode_counts = [count_modes(vec(z_all[d, 1, :])) for d in 1:q_size]

    # Score: prefer multimodal dims, break ties by z_A/z_B separation
    scores = mode_counts .* zdiff
    top_dims = sortperm(scores; rev = true)[1:num_density_dims]

    fig = Figure(size = (700, 550), backgroundcolor = :white)

    # Image row
    for col in 1:num_cols
        ax = CairoMakie.Axis(fig[2, col], aspect = DataAspect())
        hidedecorations!(ax)
        hidespines!(ax)
        image!(ax, all_rgb[col])
    end

    # Header
    Label(fig[1, 1], L"z_A", fontsize = 18, halign = :center, valign = :bottom)
    Label(fig[1, div(num_cols, 2):(div(num_cols, 2) + 1)], L"\longrightarrow", fontsize = 18, halign = :center, valign = :bottom)
    Label(fig[1, num_cols], L"z_B", fontsize = 18, halign = :center, valign = :bottom)

    # Prior density plots (empirical KDE from prior samples)
    # Dimensions selected by largest |z_A - z_B| difference
    dim_colors = [:royalblue, :firebrick, :forestgreen]
    for (di, dim) in enumerate(top_dims)
        is_last = di == num_density_dims
        ax = CairoMakie.Axis(
            fig[2 + di, 1:num_cols],
            ylabel = L"p_{f,%$dim}(z_{%$dim})",
            xlabel = is_last ? L"z_q" : "",
            xgridvisible = false,
            ygridvisible = false,
        )
        !is_last && hidexdecorations!(ax; ticks = false)
        hidespines!(ax, :t, :r)

        samples_dim = vec(z_all[dim, 1, :])
        density!(ax, samples_dim; color = (dim_colors[di], 0.15), strokecolor = (dim_colors[di], 0.8), strokewidth = 1.5)
        vlines!(ax, [z_a[dim, 1, 1]]; color = :black, linewidth = 1.5, linestyle = :solid)
        vlines!(ax, [z_b[dim, 1, 1]]; color = :black, linewidth = 1.5, linestyle = :dash)
    end

    Legend(
        fig[2 + num_density_dims + 1, 1:num_cols],
        [
            LineElement(color = :black, linewidth = 1.5, linestyle = :solid),
            LineElement(color = :black, linewidth = 1.5, linestyle = :dash),
        ],
        [L"z_A", L"z_B"],
        orientation = :horizontal,
        framevisible = false,
        labelsize = 18,
    )

    for col in 1:num_cols
        colsize!(fig.layout, col, Relative(1 / num_cols))
    end
    rowgap!(fig.layout, 4)
    rowgap!(fig.layout, 2, 12)

    save(save_dir * "slerp_$(pair_idx).png", fig, px_per_unit = 5)
    println("Saved slerp_$(pair_idx).png")
end

println("Results in $save_dir")
