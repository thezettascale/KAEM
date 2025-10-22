# using BenchmarkTools, ConfParser, Lux, Random, CUDA, ComponentArrays, CSV, DataFrames

# ENV["GPU"] = true
# ENV["FULL_QUANT"] = "FP32"
# ENV["HALF_QUANT"] = "FP32"

# include("../src/pipeline/data_utils.jl")
# using .DataUtils: get_vision_dataset

# include("../src/utils.jl")
# using .Utils

# include("../src/T-KAM/T-KAM.jl")
# using .T_KAM_model

# include("../src/T-KAM/model_setup.jl")
# using .ModelSetup

# conf = ConfParse("config/celeba_pang_config.ini")
# parse_conf!(conf)

# rng = Random.MersenneTwister(1)

# commit!(conf, "CNN", "use_cnn_lkhood", "true")
# commit!(conf, "SEQ", "sequence_length", "0")
# commit!(conf, "TRAINING", "verbose", "false")
# commit!(conf, "POST_LANGEVIN", "use_langevin", "true")
# commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")

# dataset, img_size = get_vision_dataset(
#     "CELEBAPANG",
#     parse(Int, retrieve(conf, "TRAINING", "N_train")),
#     parse(Int, retrieve(conf, "TRAINING", "N_test")),
#     parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"));
#     cnn = true,
#     img_resize = (64, 64),
# )[1:2]

# function setup_model(N_l)
#     commit!(conf, "PRIOR_LANGEVIN", "iters", "$(N_l)")
#     model = init_T_KAM(dataset, conf, img_size; rng = rng)

#     x_test, loader_state = iterate(model.train_loader)
#     x_test = pu(x_test)
#     model, ps, st_kan, st_lux = prep_model(model, x_test; rng = rng)

#     return model, half_quant.(ps), st_kan, st_lux
# end

# function benchmark_prior(model, ps, st_kan, st_lux)
#     first(model.sample_prior(model, model.grid_updates_samples, ps, st_kan, st_lux, rng))
# end


# results = DataFrame(
#     N_l = Int[],
#     time_mean = Float64[],
#     time_std = Float64[],
#     memory_estimate = Float64[],
#     allocations = Int[],
#     gc_percent = Float64[],
# )

# for N_l in [10, 20, 30, 40, 50]
#     println("Benchmarking N_l = $N_l...")

#     model, ps, st_kan, st_lux = setup_model(N_l)

#     CUDA.reclaim()
#     GC.gc()

#     b = @benchmark benchmark_prior($model, $ps, $st_kan, $st_lux)

#     push!(
#         results,
#         (
#             N_l,
#             b.times[end] / 1e9,  # Convert nanoseconds to seconds (median time)
#             std(b.times) / 1e9,  # Standard deviation
#             b.memory / (1024^3),  # Convert bytes to GiB
#             b.allocs,
#             b.gctimes[end] / b.times[end] * 100,  # GC percentage
#         ),
#     )
# end

# # Save results to CSV
# CSV.write("benches/results/prior_steps.csv", results)
# println("Results saved to prior_steps.csv")
# println(results)
