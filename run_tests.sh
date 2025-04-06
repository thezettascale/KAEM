#!/bin/bash

TEST_DIR="tests"
test_files=$(ls "$TEST_DIR"/*.jl)
LOG_FILE="test.log"
SESSION_NAME="test_session"

tmux new-session -d -s "$SESSION_NAME" bash

echo "" > "$LOG_FILE"

for test_file in $test_files; do
    echo "Running $test_file" | tee -a "$LOG_FILE"
    
    tmux send-keys -t "$SESSION_NAME" "julia --threads auto \"$test_file\" 2>&1 | tee -a \"$LOG_FILE\"" C-m
    tmux send-keys -t "$SESSION_NAME" "if [ \$? -ne 0 ]; then echo 'Test failed: $test_file' | tee -a \"$LOG_FILE\"; exit 1; fi" C-m
    
    tmux send-keys -t "$SESSION_NAME" "tmux wait-for -S test_done" C-m
    tmux wait-for test_done
done

echo "All tests passed!" | tee -a "$LOG_FILE"


