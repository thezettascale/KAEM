module DataUtils

export get_vision_dataset, get_text_dataset

include("../utils.jl")
using .Utils: pu

using MLDatasets, Embeddings, Images, ImageTransformations, HDF5, Statistics
using Flux: onehotbatch
using HuggingFaceDatasets: load_dataset

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

dataset_mapping = Dict(
    "MNIST" => MLDatasets.MNIST(),
    "FMNIST" => MLDatasets.FashionMNIST(),
    "CIFAR10" => MLDatasets.CIFAR10(),
    "SVHN" => MLDatasets.SVHN2(),
    "PTB" => MLDatasets.PTBLM(),
    "CELEBA" =>
        load_dataset("nielsr/CelebA-faces", split = "train").with_format("julia"),
)

# Huggingface datasets loading is lazy, so batch load
function batch_process(subset; img_resize::Union{Nothing, Tuple{Int, Int}} = (32, 32))
    channel_views = map(x -> channelview(x), subset)
    subdata = cat(channel_views..., dims = 4)
    return imresize(permutedims(subdata, (2, 3, 1, 4)), img_resize) ./ 255
end

function get_vision_dataset(
        dataset_name::String,
        N_train::Int,
        N_test::Int,
        num_generated_samples::Int;
        img_resize::Union{Nothing, Tuple{Int, Int}} = nothing,
        cnn::Bool = false,
        batch_size::Int = 100,
    )
    """Load and optionally resize a vision dataset. Returns (data, img_shape, save_subset)."""
    dataset = begin
        if dataset_name == "DARCY_PERM" || dataset_name == "DARCY_FLOW"
            data = h5open("PDE_data/darcy_32/darcy_train_32.h5", "r") do file
                read(file, "y")
            end
            data = data[:, :, 1:(N_train + N_test)]
            data = (data .- minimum(data)) ./ (maximum(data) - minimum(data))
            data = isnothing(img_resize) ? data : imresize(data, img_resize)
            data
        elseif dataset_name == "CELEBA"
            celeba = dataset_mapping[dataset_name]
            num_iters = fld(N_train + N_test, batch_size)
            data = zeros(Float32, img_resize..., 3, N_train + N_test)
            for i in 1:num_iters
                start_idx = (i - 1) * batch_size + 1
                end_idx = min(i * batch_size, N_train + N_test)
                data[:, :, :, start_idx:end_idx] =
                    batch_process(
                    celeba[start_idx:end_idx]["image"];
                    img_resize = img_resize,
                ) .|> Float32

                if i % 10 == 0
                    GC.gc()
                end
            end

            data
        else
            data = dataset_mapping[dataset_name][1:(N_train + N_test)].features
            data = isnothing(img_resize) ? data : imresize(data, img_resize)
            data
        end
    end

    dataset = dataset .|> Float32
    img_shape = size(dataset)[1:(end - 1)]

    img_shape =
        (
            dataset_name == "CIFAR10" ||
            dataset_name == "SVHN" ||
            dataset_name == "CELEBA"
        ) ? img_shape : (img_shape..., 1)
    dataset =
        (
            dataset_name == "CIFAR10" ||
            dataset_name == "SVHN" ||
            dataset_name == "CELEBA"
        ) ? dataset : reshape(dataset, img_shape..., :)
    save_dataset = dataset[:, :, :, 1:min(num_generated_samples, size(dataset)[end])]

    println("Resized dataset to $(img_shape)")
    return dataset, img_shape, save_dataset
end

function index_sentence(sentence::Vector{String}, max_length::Int, vocab::Dict{String, Int})
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
    """Load a text dataset with GloVe embeddings. Returns (data, shape, save_subset, vocab_len)."""
    dataset = dataset_mapping[dataset_name][1:(N_train + N_test)].features # Already tokenized
    emb = load_embeddings(GloVe) # Pre-trained embeddings

    vocab = Dict(word => i for (i, word) in enumerate(emb.vocab[1:vocab_size]))
    vocab["<pad>"] = length(vocab) + 1
    vocab["<unk>"] = length(vocab) + 1
    embedding_dim = size(emb.embeddings, 1)

    max_length = maximum(length(sentence) for sentence in dataset)
    embedding_matrix = zeros(Float32, embedding_dim, length(vocab))
    indexed_dataset =
        map(sentence -> index_sentence(sentence, sequence_length, vocab), dataset)

    dataset = hcat(indexed_dataset...)
    save_dataset = dataset[:, 1:num_generated_samples]

    return_data = zeros(Float32, length(vocab), size(dataset)...)
    num_iters = fld(size(dataset, 2), batch_size)

    # Had some issues, so batched
    for i in 1:num_iters
        start_idx = (i - 1) * batch_size + 1
        end_idx = min(i * batch_size, size(dataset, 2))
        return_data[:, :, start_idx:end_idx] =
            collect(Float32, onehotbatch(dataset[:, start_idx:end_idx], 1:length(vocab)))
    end

    return return_data, (size(return_data, 1), size(return_data, 2)), save_dataset
end

end
