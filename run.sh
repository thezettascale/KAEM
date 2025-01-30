#!/bin/bash

datasets=("MNIST" "FMNIST" "CIFAR10" "SVHN")

for dataset in "${datasets[@]}"; do
    session_name="run_$dataset"
    tmux new-session -d -s "$session_name" "DATASET=$dataset julia --threads auto main.jl > output_${dataset}.log 2>&1"
    echo "Started tmux session '$session_name' for dataset $dataset"
done