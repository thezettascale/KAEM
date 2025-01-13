#!/bin/bash

# Activate T-KAM environment
source ~/anaconda3/etc/profile.d/conda.sh
echo "Activating conda environment T-KAM..."
conda activate T-KAM
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment T-KAM"
    exit 1
fi

# Run all plotting scripts
echo "Running Python scripts in the plotting/ directory..."
find plotting/ -type f -name "*.py" | while read script; do
    echo "Running $script..."
    python "$script"
    
    # Check if the script ran successfully
    if [ $? -ne 0 ]; then
        echo "Failed to run $script"
        exit 1
    fi
done

echo "All plotting scripts executed successfully!"
