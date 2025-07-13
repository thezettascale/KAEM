#!/bin/bash

set -e

echo "Running T-KAM plotting scripts..."

PLOT_FILES=$(find plotting/ -name "*.py" -type f | sort)

if [ -z "$PLOT_FILES" ]; then
    echo "No plotting files found in plotting/ directory"
    exit 1
fi

echo "Found plotting files:"
echo "$PLOT_FILES"
echo ""

# Create figures directory if it doesn't exist
mkdir -p figures/results
mkdir -p figures/visual
mkdir -p figures/benchmark

for plot_file in $PLOT_FILES; do
    echo "Running $plot_file..."
    echo "=========================================="
    python "$plot_file"
    echo "=========================================="
    echo ""
done

echo "All plotting scripts completed!"
echo "Figures saved to figures/ directory" 