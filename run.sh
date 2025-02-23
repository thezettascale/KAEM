#!/bin/bash

# datasets=("MNIST" "FMNIST")
datasets=("SVHN" "CIFAR10")
# # # datasets=("DARCY_FLOW")

for dataset in "${datasets[@]}"; do
    session_name="IS_$dataset"
    echo "Starting $dataset Importance Sampling run..."
    
    # Runs sequentially
    tmux new-session -d -s "$session_name" "DATASET=$dataset julia --threads auto main_importance.jl > Vanilla_${dataset}.log 2>&1; tmux kill-session -t $session_name"
    
    while tmux has-session -t "$session_name" 2>/dev/null; do
        sleep 5
    done
    
    echo "$dataset Importance Sampling completed."
done

datasets=("SVHN" "CIFAR10")

for dataset in "${datasets[@]}"; do
    session_name="TI_$dataset"
    echo "Starting $dataset Thermodynamic Integration run..."
    
    # Runs sequentially
    tmux new-session -d -s "$session_name" "DATASET=$dataset julia --threads auto main_thermodynamic.jl > Thermo_${dataset}.log 2>&1; tmux kill-session -t $session_name"
    
    while tmux has-session -t "$session_name" 2>/dev/null; do
        sleep 5
    done
    
    echo "$dataset Thermodynamic Integration completed."
done
