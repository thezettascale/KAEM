#!/bin/bash

set -e

echo "Running T-KAM Julia tests..."

TEST_FILES=$(find tests/ -name "*.jl" -type f | sort)

if [ -z "$TEST_FILES" ]; then
    echo "No test files found in tests/ directory"
    exit 1
fi

echo "Found test files:"
echo "$TEST_FILES"
echo ""

# Run each test file
for test_file in $TEST_FILES; do
    echo "Running $test_file..."
    echo "=========================================="
    julia --project=. --threads=auto "$test_file"
    echo "=========================================="
    echo ""
done

echo "All tests completed!" 