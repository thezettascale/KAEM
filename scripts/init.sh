#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

set -e

echo -e "${GREEN}Setting up T-KAM development environment...${NC}"

if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Anaconda or Miniconda first:"
    echo "  - Anaconda: https://www.anaconda.com/products/distribution"
    echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

if ! command -v julia &> /dev/null; then
    echo -e "${RED}Error: julia is not installed or not in PATH${NC}"
    echo "Please install Julia first:"
    echo "  - Julia: https://julialang.org/downloads/"
    exit 1
fi

CONDA_TYPE=""
if conda info --base | grep -q "miniconda"; then
    CONDA_TYPE="miniconda"
    echo -e "${GREEN}Detected: Miniconda${NC}"
elif conda info --base | grep -q "anaconda"; then
    CONDA_TYPE="anaconda"
    echo -e "${GREEN}Detected: Anaconda${NC}"
else
    echo -e "${YELLOW}Warning: Could not determine conda type, proceeding anyway${NC}"
fi

ENV_NAME="T-KAM"

if conda env list | grep -q "$ENV_NAME"; then
    echo -e "${YELLOW}Environment '$ENV_NAME' already exists. Do you want to remove it and recreate? (y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo -e "${GREEN}Removing existing environment...${NC}"
        conda env remove -n "$ENV_NAME" -y
    else
        echo -e "${YELLOW}Using existing environment. Activating...${NC}"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
        echo -e "${GREEN}Environment activated successfully!${NC}"
        echo -e "${GREEN}Installing/updating dependencies...${NC}"
        cd "$(dirname "$0")/.."
        pip install -e ".[dev]"
        echo -e "${GREEN}Setup completed!${NC}"
        exit 0
    fi
fi

echo -e "${GREEN}Creating new conda environment '$ENV_NAME'...${NC}"
conda create -n "$ENV_NAME" python=3.11 -y

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to create conda environment${NC}"
    exit 1
fi

echo -e "${GREEN}Activating environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate conda environment${NC}"
    exit 1
fi

echo -e "${GREEN}Installing tmux for development sessions...${NC}"
conda install -c conda-forge tmux -y

echo -e "${GREEN}Installing project dependencies...${NC}"
cd "$(dirname "$0")/.."

pip install -e ".[dev]"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed successfully!${NC}"
    
    echo -e "${GREEN}Installing Julia dependencies...${NC}"
    echo -e "${YELLOW}Note: This may take a while on first run${NC}"
    julia --project=. -e "
    using Pkg
    println(\"Installing packages from Project.toml...\")
    Pkg.Registry.update()
    Pkg.instantiate()
    println(\"✓ All Julia dependencies installed successfully!\")
    "
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Julia dependencies installed successfully!${NC}"
    else
        echo -e "${RED}✗ Failed to install Julia dependencies${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Testing installation...${NC}"
    python -c "
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import h5py
import torch
import torch_fidelity
import sklearn
from PIL import Image
print('✓ All Python imports successful!')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ NumPy version: {np.__version__}')
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('✓ Using CPU (CUDA not available)')
"

    julia --project=. -e "
using Pkg
using CUDA
using Lux
using Flux
using Distributions
using Plots
using cuDNN
using NNlib
println(\"✓ All Julia imports successful!\")
println(\"✓ CUDA available: \", CUDA.functional())
    "
    
    echo -e "${GREEN}✓ Setup completed successfully!${NC}"
    echo -e "${GREEN}To activate the environment, run: conda activate $ENV_NAME${NC}"
    echo -e "${GREEN}To run tests: make test${NC}"
    echo -e "${GREEN}To start development session: make dev${NC}"
    echo -e "${GREEN}To start training: make train${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi 