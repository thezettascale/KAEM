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

function benchmark_MALA(N_l)
    commit!(conf, "THERMODYNAMIC_INTEGRATION", "N_langevin_per_temp", "$(N_l)")

    model = init_T_KAM(dataset, conf, img_size)
    ps, st = Lux.setup(Random.GLOBAL_RNG, model)
    x_test = first(model.train_loader) 

    first(gradient(p -> first(model.loss_fcn(model, p, st, x_test)), ps))
end

display(@benchmark CUDA.@sync benchmark_MALA(5))
display(@benchmark CUDA.@sync benchmark_MALA(10))
display(@benchmark CUDA.@sync benchmark_MALA(15))
display(@benchmark CUDA.@sync benchmark_MALA(20))
