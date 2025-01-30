#!/bin/bash

datasets=("MNIST" "FMNIST" "CIFAR10" "SVHN")

for dataset in "${datasets[@]}"; do
    session_name="run_$dataset"
    echo "Starting $dataset run..."
    
    tmux new-session -d -s "$session_name" "DATASET=$dataset julia --threads auto main.jl > output_${dataset}.log 2>&1; tmux kill-session -t $session_name"
    
    # Runs sequentially
    while tmux has-session -t "$session_name" 2>/dev/null; do
        sleep 5
    done
    
    echo "$dataset completed."
done