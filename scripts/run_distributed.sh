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
NUM_WORKERS="${NUM_WORKERS:-auto}"
HOSTFILE="${HOSTFILE:-}"

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

run_distributed_julia() {
    local workers=$1

    print_status "Starting distributed training with $workers worker(s)"
    print_status "Dataset: $DATASET, Mode: $MODE"

    export JULIA_NUM_THREADS=auto

    if [[ $workers -eq 1 ]]; then
        print_status "Running single-device training"
        DATASET="$DATASET" MODE="$MODE" julia --project=. --threads=auto main.jl
    else
        print_status "Running multi-device distributed training"

        if [[ -n "$HOSTFILE" ]] && [[ -f "$HOSTFILE" ]]; then
            print_status "Using hostfile: $HOSTFILE"
            DATASET="$DATASET" MODE="$MODE" julia --project=. --threads=auto \
                -p "$workers" \
                --machine-file "$HOSTFILE" \
                distributed_main.jl
        else
            print_status "Using local multi-device setup"
            DATASET="$DATASET" MODE="$MODE" julia --project=. --threads=auto \
                -p "$workers" \
                distributed_main.jl
        fi
    fi
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
    print_status "KAEM Distributed Training"
    echo "=============================================="

    if [[ ! -f "main.jl" ]]; then
        print_error "main.jl not found. Please run from project root."
        exit 1
    fi

    # Check for TPU pod first (special case)
    if is_tpu_pod; then
        print_status "Configuration:"
        print_status "  Dataset: $DATASET"
        print_status "  Mode: $MODE"
        print_status "  Device: TPU Pod (XLA SPMD)"
        echo "=============================================="
        run_tpu_pod
        print_success "TPU pod training completed"
        return
    fi

    local workers
    workers=$(get_num_workers)

    print_status "Configuration:"
    print_status "  Dataset: $DATASET"
    print_status "  Mode: $MODE"
    print_status "  Workers: $workers"

    if [[ ! -f "distributed_main.jl" ]] && [[ $workers -gt 1 ]]; then
        print_status "Creating distributed_main.jl wrapper..."
        cat > distributed_main.jl << 'JULIA_EOF'
using Distributed

if nworkers() == 1 && length(ARGS) > 0
    addprocs(parse(Int, ARGS[1]))
end

println("Running with $(nworkers()) worker(s)")

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere include("src/pipeline/trainer.jl")
@everywhere using .trainer

include("main.jl")
JULIA_EOF
        print_success "Created distributed_main.jl"
    fi

    echo "=============================================="

    run_distributed_julia "$workers"
    print_success "Distributed training completed"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --workers|-w)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --hostfile|-H)
            HOSTFILE="$2"
            shift 2
            ;;
        --mpi)
            USE_MPI=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --workers, -w NUM    Number of workers (default: auto-detect)"
            echo "  --hostfile, -H FILE  Hostfile for cluster execution"
            echo "  --mpi                Use MPI instead of Julia Distributed"
            echo "  --help, -h           Show this help"
            echo ""
            echo "Environment variables:"
            echo "  DATASET              Dataset to train on (default: MNIST)"
            echo "  MODE                 Training mode: thermo, vanilla, variational (default: thermo)"
            echo "  NUM_WORKERS          Number of workers (overridden by --workers)"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Handle MPI mode
if [[ "${USE_MPI:-false}" == "true" ]]; then
    workers=$(get_num_workers)
    run_mpi "$workers"
else
    main
fi
