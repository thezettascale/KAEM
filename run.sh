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
    fi

    echo "[$n/$total] done"
done

echo ""
echo "all $total job(s) finished"
