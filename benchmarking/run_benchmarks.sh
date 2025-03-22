#!/bin/bash

JL_FILES=("benchmarking/latent_dim.jl" "benchmarking/MALA_steps.jl")

for benchmark in $JL_FILES; 
do
    echo "Running $benchmark"
    # julia --threads auto "$benchmark" > "$benchmark.txt"

    tmux new-session -d -s "$benchmark" "julia --threads auto $benchmark > $benchmark.txt"

    while tmux has-session -t "$benchmark" 2>/dev/null; do
        sleep 5
    done

    echo "$benchmark completed!"
done

echo "All benchmarks completed!"