#!/bin/bash

echo "Running setup..."
julia "setup/requirements.jl"
python "setup/py_requirements.py"
if [ $? -ne 0 ]; then
    echo "Failed to run requirements.jl"
    exit 1
fi