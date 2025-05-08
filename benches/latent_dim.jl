using BenchmarkTools, ConfParser, Lux, Zygote, Random, CUDA, ComponentArrays

ENV["GPU"] = true
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/T-KAM/T-KAM.jl")
include("../src/ML_pipeline/data_utils.jl")
include("../src/utils.jl")
using .T_KAM_model
using .DataUtils: get_vision_dataset
using .Utils: device, half_quant

conf = ConfParse("config/nist_config.ini")
parse_conf!(conf)
commit!(conf, "MALA", "use_langevin", "false")
commit!(conf, "THERMODYNAMIC_INTEGRATION", "num_temps", "-1")
commit!(conf, "CNN", "use_cnn_lkhood", "false")
commit!(conf, "SEQ", "sequence_length", "0") 
commit!(conf, "TRAINING", "verbose", "false") 

dataset, img_size = get_vision_dataset(
    "MNIST",
    parse(Int, retrieve(conf, "TRAINING", "N_train")),
    parse(Int, retrieve(conf, "TRAINING", "N_test")),
    parse(Int, retrieve(conf, "TRAINING", "num_generated_samples"));
)[1:2]

function benchmark_dim(n_z)
    commit!(conf, "EBM_PRIOR", "layer_widths", "$(n_z), $(2*n_z+1)")
    commit!(conf, "KAN_LIKELIHOOD", "widths", "$(2*n_z+1), $(4*n_z+2)")

    model = init_T_KAM(dataset, conf, img_size)
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    model = move_to_hq(model)
    x_test = device(first(model.train_loader))
    ps, st = ComponentArray(ps) |> device, st |> device 

    first(gradient(p -> first(model.loss_fcn(model, p, st, x_test)), half_quant.(ps)))
end

display(@benchmark CUDA.@sync benchmark_dim(10))
display(@benchmark CUDA.@sync benchmark_dim(20))
display(@benchmark CUDA.@sync benchmark_dim(30))
display(@benchmark CUDA.@sync benchmark_dim(40))
display(@benchmark CUDA.@sync benchmark_dim(50))
