#!/bin/bash

echo "Running T-KAM tests..."

TEST_FILES=$(find tests/ -name "*.jl" -type f | sort)

if [ -z "$TEST_FILES" ]; then
    echo "No test files found in tests/ directory"
    exit 1
fi

echo "Found test files:"
echo "$TEST_FILES"
echo ""

PASSED_TESTS=()
FAILED_TESTS=()

for test_file in $TEST_FILES; do
    echo "Running $test_file..."
    echo "=========================================="
    
    if julia --project=. --threads=auto "$test_file"; then
        echo "✓ PASSED: $test_file"
        PASSED_TESTS+=("$test_file")
    else
        echo "✗ FAILED: $test_file"
        FAILED_TESTS+=("$test_file")
    fi
    
    echo "=========================================="
    echo ""
done

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Passed: ${#PASSED_TESTS[@]}"
echo "Failed: ${#FAILED_TESTS[@]}"
echo ""

if [ ${#PASSED_TESTS[@]} -gt 0 ]; then
    echo "✓ PASSED TESTS:"
    for test in "${PASSED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
fi

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "✗ FAILED TESTS:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
fi

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "Some tests failed. Exiting with error code 1."
    exit 1
else
    echo "All tests passed!"
    exit 0
fi 