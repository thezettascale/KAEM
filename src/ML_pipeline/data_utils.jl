
module DataUtils

export get_vision_dataset, get_text_dataset

include("../utils.jl")
using .Utils: device, half_quant

using MLDatasets, Embeddings

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

dataset_mapping = Dict(
    "MNIST" => MLDatasets.MNIST(),
    "FMNIST" => MLDatasets.FashionMNIST(),
    "CIFAR10" => MLDatasets.CIFAR10(),
    "SVHN" => MLDatasets.SVHN2(),
    "PTB" => MLDatasets.PTBLM(),
    # "UD" => MLDatasets.UD_English(),
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

function embed_sentence(
    sentence::Vector{String},
    max_length::Int,
    vocab::Dict{String, Int},
    embedding_matrix::Array{half_quant, 2},
    embedding_dim::Int
    )
    embedded = zeros(half_quant, embedding_dim, max_length, 1)
    for (i, token) in enumerate(sentence[1:min(length(sentence), max_length)])
        if token in keys(vocab)
            embedded[:, i, 1] = embedding_matrix[:, vocab[token]]
        end
    end
    return embedded
end

function pad_sequence(
    seq::Array{half_quant, 3}, 
    max_length::Int,
    embedding_dim::Int
    )

    if size(seq, 2) < max_length
        return hcat(seq, zeros(half_quant, embedding_dim, max_length - size(seq, 2), 1))
    else
        return seq[:, 1:max_length, :]
    end
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
        The shape of the dataset.
        The dataset to save.
    """
    dataset = dataset_mapping[dataset_name][1:N_train+N_test].features # Already tokenized
    glove = load_embeddings(GloVe) # Pre-trained embeddings
    vocab = Dict(word => i for (i, word) in enumerate(glove.vocab))
    
    embedding_dim = size(glove.embeddings, 1)
    max_length = maximum(length(sentence) for sentence in dataset)
    embedding_matrix = zeros(half_quant, embedding_dim, length(vocab))

    for (word, i) in vocab
        embedding_matrix[:, i] = glove.embeddings[:, i] .|> half_quant
    end

    embed_dataset = map(sentence -> embed_sentence(sentence, max_length, vocab, embedding_matrix, embedding_dim), dataset)
    dataset = reduce((x,y) -> cat(x, y, dims=3), map(seq -> pad_sequence(seq, max_length, embedding_dim), embed_dataset))

    save_dataset = dataset[:, :, 1:num_generated_samples]
    return dataset, size(dataset)[1:end-1], save_dataset
end 

end