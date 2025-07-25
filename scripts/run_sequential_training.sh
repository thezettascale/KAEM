#!/bin/bash

set -e  # Exit on any error

CONFIG_FILE="${1:-jobs.txt}"
SESSION_NAME="tkam_sequential"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

session_exists() {
    tmux has-session -t "$1" 2>/dev/null
}

# Check periodically and wait for tmux session to complete
wait_for_training_completion() {
    print_status "Waiting for current training session to complete..."
    
    while session_exists "tkam_train"; do
        sleep 10
    done
    
    print_success "Training session completed"
}

run_training_job() {
    local dataset="$1"
    local mode="$2"
    local job_num="$3"
    local total_jobs="$4"
    
    echo
    echo "============================================================"
    print_status "Starting Job $job_num/$total_jobs: $dataset - $mode"
    echo "============================================================"
    
    # Kill existing sessions
    if session_exists "tkam_train"; then
        print_warning "Killing existing training session."
        tmux kill-session -t tkam_train 2>/dev/null || true
        sleep 2
    fi
    
    print_status "Running: make train DATASET=$dataset MODE=$mode"
    
    make train DATASET="$dataset" MODE="$mode"
    
    wait_for_training_completion
    
    print_success "Job $job_num/$total_jobs completed: $dataset - $mode"
}

load_config() {
    local config_file="$1"
    
    if [[ ! -f "$config_file" ]]; then
        print_error "Configuration file '$config_file' not found"
        exit 1
    fi
    
    local jobs=()
    local line_num=0
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        ((line_num++))
        
        # Skip comments and empty lines
        if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]]; then
            continue
        fi
        
        # Parse dataset and thermo/vanilla
        read -r dataset mode <<< "$line"
        
        if [[ -z "$dataset" || -z "$mode" ]]; then
            print_warning "Invalid line $line_num: '$line' (skipping)"
            continue
        fi
        
        # Checks
        case "$dataset" in
            MNIST|FMNIST|CIFAR10|SVHN|PTB|SMS_SPAM|DARCY_PERM|DARCY_FLOW)
                ;;
            *)
                print_warning "Unknown dataset '$dataset' on line $line_num (skipping)"
                continue
                ;;
        esac
        
        case "$mode" in
            thermo|vanilla)
                ;;
            *)
                print_warning "Unknown mode '$mode' on line $line_num (skipping)"
                continue
                ;;
        esac
        
        jobs+=("$dataset $mode")
    done < "$config_file"
    
    echo "${jobs[@]}"
}
    
main() {
    print_status "Sequential Runner"
    print_status "Configuration file: $CONFIG_FILE"
    
    local jobs
    mapfile -t jobs < <(load_config "$CONFIG_FILE")
    
    local total_jobs=${#jobs[@]}
    
    if [[ $total_jobs -eq 0 ]]; then
        print_error "No valid jobs found in configuration file"
        exit 1
    fi
    
    print_status "Found $total_jobs training jobs to run sequentially"
    
    # Signal handler
    trap 'print_warning "Interrupted by user. Stopping training sequence."; tmux kill-session -t tkam_train 2>/dev/null || true; exit 0' INT TERM
    
    for i in "${!jobs[@]}"; do
        local job_num=$((i + 1))
        read -r dataset mode <<< "${jobs[i]}"
        
        print_status "Preparing job $job_num/$total_jobs: $dataset - $mode"
        
        run_training_job "$dataset" "$mode" "$job_num" "$total_jobs"
        
        sleep 5
    done
    
    echo
    echo "============================================================"
    print_success "All training jobs completed."
    print_status "$total_jobs jobs finished"
    echo "============================================================"
}

if [[ ! -f "Makefile" ]]; then
    print_error "Makefile not found. Please run this script from the project root directory."
    exit 1
fi

if ! command -v tmux &> /dev/null; then
    print_error "tmux is not installed. Please install tmux to use this script."
    exit 1
fi

main "$@" 