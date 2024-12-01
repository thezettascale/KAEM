#!/bin/bash

# Creates conda environment
source ~/anaconda3/etc/profile.d/conda.sh

conda env list | grep -q 'LV-KAM'
if [ $? -ne 0 ]; then
    echo "Creating conda environment LV-KAM..."
    conda create -n LV-KAM python=3.11 -y  
fi

conda activate LV-KAM

# Install Python requirements
echo "Installing Python requirements..."
python setup/python_reqs.py
if [ $? -ne 0 ]; then
    echo "Failed to install Python requirements"
    exit 1
fi

# Install Julia requirements
echo "Running Julia setup..."
julia setup/julia_reqs.jl
if [ $? -ne 0 ]; then
    echo "Failed to run requirements.jl"
    exit 1
fi

echo "Setup complete!"
