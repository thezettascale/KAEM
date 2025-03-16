using BenchmarkTools, ConfParser, Lux, Zygote, Random, CUDA

ENV["GPU"] = false
ENV["FULL_QUANT"] = "FP32"
ENV["HALF_QUANT"] = "FP32"

include("../src/T-KAM/T-KAM.jl")
include("../src/ML_pipeline/data_utils.jl")
using .T_KAM_model
using .DataUtils: get_vision_dataset

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
    img_resize=(28, 28),
    cnn=false
)[1:2]

function benchmark_dim(n_z)
    commit!(conf, "EBM_PRIOR", "layer_widths", "$(n_z), $(2*n_z+1)")
    commit!(conf, "KAN_LIKELIHOOD", "widths", "$(2*n_z+1), $(4*n_z+2)")

    model = init_T_KAM(dataset, conf, img_size)
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    x_test = first(model.train_loader) 

    first(gradient(p -> first(model.loss_fcn(model, p, st, x_test)), ps))
end

display(@benchmark CUDA.@sync benchmark_dim(25))
display(@benchmark CUDA.@sync benchmark_dim(50))
display(@benchmark CUDA.@sync benchmark_dim(100))