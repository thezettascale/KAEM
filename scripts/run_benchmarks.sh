#!/bin/bash

set -e

echo "Running KAEM Julia benchmarks..."

BENCH_FILES=$(find benches/ -name "*.jl" -type f | sort)

if [ -z "$BENCH_FILES" ]; then
    echo "No benchmark files found in benches/ directory"
    exit 1
fi

echo "Found benchmark files:"
echo "$BENCH_FILES"
echo ""

mkdir -p benches/results

for bench_file in $BENCH_FILES; do
    echo "Running $bench_file..."
    echo "=========================================="
    julia --project=. --threads=auto "$bench_file"
    echo "=========================================="
    echo ""
done

echo "All benchmarks completed!"
echo "Results saved to benches/results/" 