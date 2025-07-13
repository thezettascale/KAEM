
module DataUtils

export get_vision_dataset, get_text_dataset

include("../utils.jl")
using .Utils: device, full_quant

using MLDatasets, Embeddings, Images, ImageTransformations, HDF5
using Flux: onehotbatch

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

dataset_mapping = Dict(
    "MNIST" => MLDatasets.MNIST(),
    "FMNIST" => MLDatasets.FashionMNIST(),
    "CIFAR10" => MLDatasets.CIFAR10(),
    "SVHN" => MLDatasets.SVHN2(),
    "PTB" => MLDatasets.PTBLM(),
    # "UD_ENGLISH" => MLDatasets.UD_English(),
    "DARCY_PERM" => h5open("PDE_data/darcy_32/darcy_train_32.h5")["x"],
    "DARCY_FLOW" => h5open("PDE_data/darcy_32/darcy_train_32.h5")["y"],
)

function get_vision_dataset(
    dataset_name::String,
    N_train::Int,
    N_test::Int,
    num_generated_samples::Int;
    img_resize::Union{Nothing,Tuple{Int,Int}} = nothing,
    cnn::Bool = false,
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
    dataset =
        (dataset_name == "DARCY_PERM" || dataset_name == "DARCY_FLOW") ?
        dataset_mapping[dataset_name][:, :, 1:(N_train+N_test)] :
        dataset_mapping[dataset_name][1:(N_train+N_test)].features
    dataset =
        (dataset_name == "DARCY_PERM" || dataset_name == "DARCY_FLOW") ?
        (dataset .- minimum(dataset)) ./ (maximum(dataset) - minimum(dataset)) : dataset
    dataset = isnothing(img_resize) ? dataset : imresize(dataset, img_resize)
    dataset = dataset .|> full_quant
    img_shape = size(dataset)[1:(end-1)]

    img_shape =
        (dataset_name == "CIFAR10" || dataset_name == "SVHN") ? img_shape :
        (img_shape..., 1)
    dataset =
        (dataset_name == "CIFAR10" || dataset_name == "SVHN") ? dataset :
        reshape(dataset, img_shape..., :)
    save_dataset = dataset[:, :, :, 1:min(num_generated_samples, size(dataset)[end])]

    println("Resized dataset to $(img_shape)")
    return dataset, img_shape, save_dataset
end

function index_sentence(sentence::Vector{String}, max_length::Int, vocab::Dict{String,Int})
    indexed = fill(vocab["<pad>"], max_length, 1)
    for (i, token) in enumerate(sentence[1:min(length(sentence), max_length)])
        if token in keys(vocab)
            indexed[i, 1] = vocab[token]
        else
            indexed[i, 1] = vocab["<unk>"] # MLDatasets already has this, but incl for completeness
        end
    end
    return indexed
end
function get_text_dataset(
    dataset_name::String,
    N_train::Int,
    N_test::Int,
    num_generated_samples::Int;
    sequence_length::Int = 100,
    vocab_size::Int = 1000,
    batch_size::Int = 100,
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
        The shape of the dataset.
        The dataset to save.
        The length of the vocabulary.
    """
    dataset = dataset_mapping[dataset_name][1:(N_train+N_test)].features # Already tokenized
    emb = load_embeddings(GloVe) # Pre-trained embeddings

    vocab = Dict(word => i for (i, word) in enumerate(emb.vocab[1:vocab_size]))
    vocab["<pad>"] = length(vocab) + 1
    vocab["<unk>"] = length(vocab) + 1
    embedding_dim = size(emb.embeddings, 1)

    max_length = maximum(length(sentence) for sentence in dataset)
    embedding_matrix = zeros(full_quant, embedding_dim, length(vocab))
    indexed_dataset =
        map(sentence -> index_sentence(sentence, sequence_length, vocab), dataset)

    dataset = reduce(hcat, indexed_dataset)
    save_dataset = dataset[:, 1:num_generated_samples]

    return_data = zeros(full_quant, length(vocab), size(dataset)...)
    num_iters = fld(size(dataset, 2), batch_size)

    # Had some issues, so batched
    for i = 1:num_iters
        start_idx = (i - 1) * batch_size + 1
        end_idx = min(i * batch_size, size(dataset, 2))
        return_data[:, :, start_idx:end_idx] =
            collect(full_quant, onehotbatch(dataset[:, start_idx:end_idx], 1:length(vocab)))
    end

    return return_data, (size(return_data, 1), size(return_data, 2)), save_dataset
end

end
