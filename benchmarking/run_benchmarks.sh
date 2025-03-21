#!/bin/bash

DIR="benchmarking"
JL_FILES=("benchmarking/latent_dim.jl" "benchmarking/MALA_steps.jl")

for benchmark in $JL_FILES; 
do
    echo "Running $benchmark"
    julia --threads auto "$benchmark" > "$benchmark.txt"
    if [ $? -ne 0 ]; then
        echo "Benchmark failed: $benchmark"
        exit 1
    fi
done

echo "All benchmarks completed!"