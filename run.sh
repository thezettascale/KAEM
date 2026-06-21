#!/bin/bash
set -e

CONFIG="${1:-jobs.txt}"

if [[ ! -f "$CONFIG" ]]; then
    echo "error: $CONFIG not found"
    exit 1
fi

mkdir -p logs
LOGFILE="logs/run_$(date +%Y%m%d_%H%M%S).log"

jobs=()
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]] && continue
    jobs+=("$line")
done < "$CONFIG"

total=${#jobs[@]}
if [[ $total -eq 0 ]]; then
    echo "no jobs found in $CONFIG"
    exit 1
fi

echo "running $total job(s) from $CONFIG"
echo "log: $LOGFILE"

# Datasets supported by / latent_interpolation.jl
LATENT_DATASETS=("CELEBA" "SVHN" "CIFAR10")

is_latent_supported() {
    local d="$1"
    for x in "${LATENT_DATASETS[@]}"; do [[ "$x" == "$d" ]] && return 0; done
    return 1
}

for i in "${!jobs[@]}"; do
    read -r dataset mode <<< "${jobs[$i]}"
    n=$((i + 1))

    echo ""
    echo "[$n/$total] $dataset $mode"

    if [[ "$mode" == "tune" ]]; then
        DATASET="$dataset" julia --project=. --threads=1 tuning.jl 2>&1 | tee -a "$LOGFILE"
    elif [[ "$mode" == baseline-* ]]; then
        model="${mode#baseline-}"
        JULIA_CONDAPKG_OFFLINE=yes MODEL="$model" DATASET="$dataset" \
            julia --project=. --threads=auto baseline.jl 2>&1 | tee -a "$LOGFILE"
    else
        DATASET="$dataset" MODE="$mode" \
            julia --project=. --threads=auto main.jl 2>&1 | tee -a "$LOGFILE"

        if [[ "$mode" == "vanilla" || "$mode" == "thermo" ]] && is_latent_supported "$dataset"; then
            echo "[$n/$total] latent_interpolation.jl $dataset $mode"
            julia --project=. --threads=auto latent_interpolation.jl "$dataset" "$mode" 2>&1 | tee -a "$LOGFILE"
        fi
    fi

    echo "[$n/$total] done"
done

echo ""
echo "all $total job(s) finished"
