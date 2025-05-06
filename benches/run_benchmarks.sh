#!/bin/bash

JL_FILES=("benches/latent_dim.jl")

for benchmark in "${JL_FILES[@]}"; do

    echo "Running $benchmark"

    tmux new-session -d -s "$benchmark" "julia --threads auto $benchmark > $benchmark.txt 2>&1; tmux kill-session -t $session_name"

    while tmux has-session -t "$benchmark" 2>/dev/null; do
        sleep 5
    done

    echo "$benchmark completed!"
done

echo "All benchmarks completed!"