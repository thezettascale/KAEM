#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

DATASET="${DATASET:-MNIST}"
MODE="${MODE:-thermo}"
MODEL="${MODEL:-}"  # For baseline modes: vae, gan, ddpm
NUM_WORKERS="${NUM_WORKERS:-auto}"
HOSTFILE="${HOSTFILE:-}"

# Check if this is a baseline job
is_baseline_mode() {
    [[ "$MODE" == baseline-* ]] || [[ -n "$MODEL" ]]
}

# Extract model from baseline-* mode
get_baseline_model() {
    if [[ -n "$MODEL" ]]; then
        echo "$MODEL"
    elif [[ "$MODE" == baseline-* ]]; then
        echo "${MODE#baseline-}"
    else
        echo ""
    fi
}

is_tpu_pod() {
    [[ -n "${TPU_WORKER_HOSTNAMES:-}" ]] || [[ -n "${TPU_CHIPS_PER_HOST_BOUNDS:-}" ]]
}

detect_devices() {
    local num_gpus=0
    local num_tpus=0

    # Check for GPUs
    if command -v nvidia-smi &> /dev/null; then
        num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    fi

    # Check for TPU pod (multi-host)
    if is_tpu_pod; then
        print_status "TPU pod detected - XLA will handle distribution"
        num_tpus=-1  # Signal TPU pod mode
        echo "$num_gpus $num_tpus"
        return
    fi

    # Check for single TPU VM
    if [[ -d "/dev/accel" ]]; then
        num_tpus=$(ls /dev/accel* 2>/dev/null | wc -l || echo 0)
    fi

    # Check for TPU VM environment
    if [[ -n "${TPU_NAME:-}" ]] || [[ -f "/usr/share/tpu/tpu-env" ]]; then
        num_tpus=1  # TPU VM typically exposes as single device with multiple cores
    fi

    echo "$num_gpus $num_tpus"
}

get_num_workers() {
    if [[ "$NUM_WORKERS" == "auto" ]]; then
        read -r num_gpus num_tpus <<< "$(detect_devices)"

        if [[ $num_tpus -gt 0 ]]; then
            print_status "Detected TPU environment"
            echo "$num_tpus"
        elif [[ $num_gpus -gt 0 ]]; then
            print_status "Detected $num_gpus GPU(s)"
            echo "$num_gpus"
        else
            print_warning "No accelerators detected, using single CPU worker"
            echo "1"
        fi
    else
        echo "$NUM_WORKERS"
    fi
}

# XLA handles distribution via SPMD
run_tpu_pod() {
    print_status "Running on TPU pod - XLA SPMD handles distribution"
    print_status "TPU_WORKER_HOSTNAMES: ${TPU_WORKER_HOSTNAMES:-not set}"

    export JULIA_NUM_THREADS=auto
    DATASET="$DATASET" MODE="$MODE" julia --project=. --threads=auto main.jl
}

run_baseline_julia() {
    local model=$1
    local device=${2:-0}

    print_status "Starting baseline training: $model on $DATASET (device $device)"

    export JULIA_NUM_THREADS=auto
    export CUDA_VISIBLE_DEVICES="$device"

    MODEL="$model" DATASET="$DATASET" julia --project=. --threads=auto baseline.jl
}

run_distributed_julia() {
    local device=${1:-0}

    # Check if baseline mode
    if is_baseline_mode; then
        local model
        model=$(get_baseline_model)
        run_baseline_julia "$model" "$device"
        return
    fi

    print_status "Starting KAEM training: $DATASET $MODE (device $device)"

    export JULIA_NUM_THREADS=auto
    export CUDA_VISIBLE_DEVICES="$device"

    DATASET="$DATASET" MODE="$MODE" julia --project=. --threads=auto main.jl
}

# Run with MPI
run_mpi() {
    local workers=$1

    print_status "Starting MPI distributed training with $workers processes"

    if ! command -v mpiexec &> /dev/null; then
        print_error "MPI not found. Please install MPI or use Julia distributed mode."
        exit 1
    fi

    if [[ -n "$HOSTFILE" ]] && [[ -f "$HOSTFILE" ]]; then
        mpiexec -n "$workers" --hostfile "$HOSTFILE" \
            julia --project=. --threads=auto mpi_main.jl
    else
        mpiexec -n "$workers" julia --project=. --threads=auto mpi_main.jl
    fi
}

main() {
    if is_baseline_mode; then
        print_status "Baseline Distributed Training"
    else
        print_status "KAEM Distributed Training"
    fi
    echo "=============================================="

    if [[ ! -f "main.jl" ]]; then
        print_error "main.jl not found. Please run from project root."
        exit 1
    fi

    # Check for TPU pod first (special case) - KAEM only
    if is_tpu_pod && ! is_baseline_mode; then
        print_status "Configuration:"
        print_status "  Dataset: $DATASET"
        print_status "  Mode: $MODE"
        print_status "  Device: TPU Pod (XLA SPMD)"
        echo "=============================================="
        run_tpu_pod
        print_success "TPU pod training completed"
        return
    fi

    local device="${DEVICE:-0}"

    print_status "Configuration:"
    print_status "  Dataset: $DATASET"
    if is_baseline_mode; then
        print_status "  Model: $(get_baseline_model)"
        print_status "  Type: Baseline"
    else
        print_status "  Mode: $MODE"
        print_status "  Type: KAEM"
    fi
    print_status "  Device: $device"

    echo "=============================================="

    run_distributed_julia "$device"
    if is_baseline_mode; then
        print_success "Baseline training completed"
    else
        print_success "KAEM training completed"
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --device|-d)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Runs a single training job. Use 'make batch' to run multiple jobs in parallel."
            echo ""
            echo "Options:"
            echo "  --device, -d NUM     GPU device index (default: 0)"
            echo "  --help, -h           Show this help"
            echo ""
            echo "Environment variables:"
            echo "  DATASET              Dataset to train on (default: MNIST)"
            echo "  MODE                 Training mode (default: thermo)"
            echo "                       KAEM: thermo, vanilla, variational"
            echo "                       Baseline: baseline-vae, baseline-gan, baseline-ddpm, baseline-pang"
            echo "  MODEL                Baseline model (alternative to MODE=baseline-*)"
            echo "  DEVICE               GPU device index (default: 0)"
            echo ""
            echo "Examples:"
            echo "  DATASET=CIFAR10 MODE=thermo $0           # KAEM training on device 0"
            echo "  DATASET=CIFAR10 MODE=baseline-vae $0     # VAE baseline on device 0"
            echo "  DEVICE=1 MODEL=gan DATASET=MNIST $0      # GAN baseline on device 1"
            echo ""
            echo "For parallel execution of multiple jobs, use:"
            echo "  make batch CONFIG=jobs.txt NUM_DEVICES=4"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
