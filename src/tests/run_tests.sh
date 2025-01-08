#!/bin/bash

TEST_DIR="src/tests"
test_files=$(ls "$TEST_DIR"/*.jl)

for test_file in $test_files; 
do
    echo "Running $test_file"
    julia "$test_file"
    if [ $? -ne 0 ]; then
        echo "Test failed: $test_file"
        exit 1
    fi
done

echo "All tests passed!"

