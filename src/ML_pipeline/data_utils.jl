
module DataUtils

export get_vision_dataset

include("../utils.jl")
using .Utils: device, half_quant

using MLDatasets, Embeddings

dataset_mapping = Dict(
    "MNIST" => MLDatasets.MNIST(),
    "FMNIST" => MLDatasets.FashionMNIST(),
    "CIFAR10" => MLDatasets.CIFAR10(),
    "SVHN" => MLDatasets.SVHN2(),
    "PTB" => MLDatasets.PTBLM(),
)

function get_vision_dataset(
    dataset_name::String,
    N_train::Int,
    N_test::Int,
    num_generated_samples::Int;
    img_resize::Union{Nothing, Tuple{Int, Int}}=nothing,
    cnn::Bool=false
    )
    """
    Load a vision dataset and resize it if necessary.

    Args:
        dataset_name: The name of the dataset.
        N_train: The number of training samples.
        N_test: The number of test samples.
        num_generated_samples: The number of samples to generate.
        img_resize: The size to resize the images to.
        cnn: A boolean indicating if the dataset is for a CNN.

    Returns:
        The dataset.
        The shape of the images.
        The dataset to save.
    """
    dataset = dataset_mapping[dataset_name][1:N_train+N_test].features # Already normalized
    dataset = isnothing(img_resize) ? dataset : imresize(dataset, img_resize)
    img_shape = size(dataset)[1:end-1]
    
    dataset = cnn ? dataset : reshape(dataset, prod(size(dataset)[1:end-1]), size(dataset)[end])
    dataset = dataset .|> half_quant
    save_dataset = (
        cnn ? 
        dataset[:,:,:,1:num_generated_samples] 
        : reshape(dataset[:, 1:num_generated_samples], img_shape..., num_generated_samples)
    )
    
    println("Resized dataset to $(img_shape)")
    return dataset, img_shape, save_dataset
end

function get_text_dataset(
    dataset_name::String,
    N_train::Int,
    N_test::Int,
    num_generated_samples::Int
    )
    """
    Load a text dataset.

    Args:
        dataset_name: The name of the dataset.
        N_train: The number of training samples.
        N_test: The number of test samples.
        num_generated_samples: The number of samples to generate.

    Returns:
        The dataset.
    """
    dataset = dataset_mapping[dataset_name][1:N_train+N_test].features # Already tokenized
    glove = load_embeddings(GloVe) # Pre-trained embeddings
    dataset = dataset .|> half_quant
end 

end