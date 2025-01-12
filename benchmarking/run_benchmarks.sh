#!/bin/bash

DIR="benchmarking"
JL_FILES=$(ls "$DIR"/*.jl)

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