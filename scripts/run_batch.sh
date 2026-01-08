#!/bin/bash

set -e

CONFIG_FILE="${1:-jobs.txt}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }

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
            thermo|vanilla|variational|tune) ;;
            *) print_warning "Unknown mode '$mode' on line $line_num (skipping)"; continue ;;
        esac

        echo "$dataset $mode"
    done < "$config_file"
}

run_distributed_job() {
    local dataset="$1"
    local mode="$2"
    local job_num="$3"
    local total_jobs="$4"

    echo
    echo "============================================================"
    print_status "Job $job_num/$total_jobs: $dataset - $mode (distributed)"
    echo "============================================================"

    DATASET="$dataset" MODE="$mode" ./scripts/run_distributed.sh

    print_success "Job $job_num/$total_jobs completed: $dataset - $mode"
}

main() {
    print_status "Distributed Sequential Runner"
    print_status "Configuration file: $CONFIG_FILE"
    print_status "NUM_WORKERS: ${NUM_WORKERS:-auto}"

    local config_output
    config_output=$(load_config "$CONFIG_FILE")
    mapfile -t jobs <<< "$config_output"

    local total_jobs=${#jobs[@]}

    if [[ $total_jobs -eq 0 ]]; then
        print_error "No valid jobs found in configuration file"
        exit 1
    fi

    print_status "Found $total_jobs jobs to run with distributed execution"

    trap 'print_warning "Interrupted. Stopping."; exit 0' INT TERM

    for i in "${!jobs[@]}"; do
        local job_num=$((i + 1))
        read -r dataset mode <<< "${jobs[i]}"
        run_distributed_job "$dataset" "$mode" "$job_num" "$total_jobs"
        sleep 5
    done

    echo
    echo "============================================================"
    print_success "All $total_jobs distributed jobs completed."
    echo "============================================================"
}

[[ ! -f "Makefile" ]] && print_error "Run from project root." && exit 1

main "$@"
