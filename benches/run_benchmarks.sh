#!/bin/bash

JL_FILES=("benches/latent_dim.jl" "benches/temperatures.jl" "benches/prior_steps.jl" "benches/ITS_single.jl")

for benchmark in "${JL_FILES[@]}"; do
    echo "Running $benchmark"
    
    julia --threads auto "$benchmark" > "$benchmark.log" 2>&1
    
    echo "$benchmark completed!"
done

echo "All benchmarks completed!"