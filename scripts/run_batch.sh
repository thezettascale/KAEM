#!/bin/bash

set -e

CONFIG_FILE="${1:-jobs.txt}"
NUM_WORKERS="${NUM_WORKERS:-auto}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }

detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0
    else
        echo 0
    fi
}

get_num_devices() {
    if [[ "$NUM_WORKERS" == "auto" ]]; then
        local num_gpus
        num_gpus=$(detect_gpus)
        if [[ $num_gpus -gt 0 ]]; then
            echo "$num_gpus"
        else
            echo 1
        fi
    else
        echo "$NUM_WORKERS"
    fi
}

load_config() {
    local config_file="$1"

    if [[ ! -f "$config_file" ]]; then
        print_error "Configuration file '$config_file' not found"
        exit 1
    fi

    local line_num=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        ((line_num++))

        [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]] && continue

        read -r dataset mode <<< "$line"

        if [[ -z "$dataset" || -z "$mode" ]]; then
            print_warning "Invalid line $line_num: '$line' (skipping)"
            continue
        fi

        case "$dataset" in
            MNIST|FMNIST|CIFAR10|SVHN|CIFAR10PANG|SVHNPANG|PTB|SMS_SPAM|DARCY_PERM|DARCY_FLOW|CELEBA|CELEBAPANG) ;;
            *) print_warning "Unknown dataset '$dataset' on line $line_num (skipping)"; continue ;;
        esac

        case "$mode" in
            thermo|vanilla|variational|tune|baseline-vae|baseline-gan|baseline-ddpm|baseline-pang) ;;
            *) print_warning "Unknown mode '$mode' on line $line_num (skipping)"; continue ;;
        esac

        echo "$dataset $mode"
    done < "$config_file"
}

run_job_on_device() {
    local dataset="$1"
    local mode="$2"
    local device="$3"
    local job_id="$4"
    local log_file="logs/batch_job_${job_id}_${dataset}_${mode}.log"

    print_status "Starting job $job_id: $dataset $mode on device $device"

    export CUDA_VISIBLE_DEVICES="$device"
    export JULIA_NUM_THREADS=auto

    if [[ "$mode" == baseline-* ]]; then
        local model="${mode#baseline-}"
        MODEL="$model" DATASET="$dataset" julia --project=. --threads=auto baseline.jl > "$log_file" 2>&1
    else
        DATASET="$dataset" MODE="$mode" julia --project=. --threads=auto main.jl > "$log_file" 2>&1
    fi
}

main() {
    print_status "Parallel Batch Runner"
    print_status "Configuration file: $CONFIG_FILE"

    local num_devices
    num_devices=$(get_num_devices)
    print_status "Using $num_devices device(s) for parallel execution"

    mkdir -p logs

    local config_output
    config_output=$(load_config "$CONFIG_FILE")
    mapfile -t jobs <<< "$config_output"

    local total_jobs=${#jobs[@]}

    if [[ $total_jobs -eq 0 ]]; then
        print_error "No valid jobs found in configuration file"
        exit 1
    fi

    print_status "Found $total_jobs jobs to distribute across $num_devices device(s)"
    echo "============================================================"

    # Track background PIDs
    declare -a pids=()
    declare -a job_info=()
    local job_idx=0

    trap 'print_warning "Interrupted. Killing background jobs..."; for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; exit 1' INT TERM

    while [[ $job_idx -lt $total_jobs ]]; do
        # Launch jobs up to num_devices in parallel
        pids=()
        job_info=()

        for ((device=0; device<num_devices && job_idx<total_jobs; device++, job_idx++)); do
            read -r dataset mode <<< "${jobs[$job_idx]}"
            local job_num=$((job_idx + 1))

            run_job_on_device "$dataset" "$mode" "$device" "$job_num" &
            pids+=($!)
            job_info+=("$job_num:$dataset:$mode:$device")
        done

        # Wait for current batch to complete
        print_status "Waiting for batch of ${#pids[@]} job(s) to complete..."

        local failed=0
        for i in "${!pids[@]}"; do
            local pid="${pids[$i]}"
            local info="${job_info[$i]}"
            IFS=':' read -r jnum ds md dev <<< "$info"

            if wait "$pid"; then
                print_success "Job $jnum completed: $ds $md (device $dev)"
            else
                print_error "Job $jnum failed: $ds $md (device $dev) - check logs/batch_job_${jnum}_${ds}_${md}.log"
                failed=$((failed + 1))
            fi
        done

        if [[ $failed -gt 0 ]]; then
            print_warning "$failed job(s) failed in this batch"
        fi

        echo "------------------------------------------------------------"
    done

    echo "============================================================"
    print_success "All $total_jobs jobs completed."
    echo "============================================================"
}

[[ ! -f "Makefile" ]] && print_error "Run from project root." && exit 1

main "$@"
