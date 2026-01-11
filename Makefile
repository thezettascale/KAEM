.PHONY: install uninstall clean test bench train train-thermo train-vanilla train-variational tune sequential distributed batch plot plot-results extract-symbolic format lint logs clear-logs julia-setup help baseline baseline-vae baseline-gan baseline-ddpm baseline-pang baseline-all

ENV_NAME = KAEM
CONDA_BASE := $(shell conda info --base 2>/dev/null || echo "")
CONDA_ACTIVATE := $(shell if [ -f "$(CONDA_BASE)/etc/profile.d/conda.sh" ]; then echo "$(CONDA_BASE)/etc/profile.d/conda.sh"; elif [ -f "$(CONDA_BASE)/Scripts/activate" ]; then echo "$(CONDA_BASE)/Scripts/activate"; else echo ""; fi)

DATASET ?= MNIST
MODE ?= thermo
NUM_DEVICES ?= auto
MODEL ?= vae
DEVICE ?= 0
CONFIG ?= jobs.txt

XLA_REACTANT_GPU_MEM_FRACTION ?= 0.9
XLA_REACTANT_GPU_PREALLOCATE ?= true
TF_GPU_ALLOCATOR ?= cuda_malloc_async
XLA_FLAGS ?=

export DATASET
export MODE
export NUM_DEVICES
export MODEL
export DEVICE
export XLA_REACTANT_GPU_MEM_FRACTION
export XLA_REACTANT_GPU_PREALLOCATE
export TF_GPU_ALLOCATOR
export XLA_FLAGS

help:
	@echo "Available targets:"
	@echo "  install       - Set up conda environment and install dependencies"
	@echo "  uninstall     - Remove only the dev environment (Conda env and Julia Manifest.toml)"
	@echo "  clean         - Remove conda environment"
	@echo "  test          - Run tests in tmux session with logging"
	@echo "  bench         - Run benchmarks in tmux session with logging"
	@echo "  train         - Start training (use: make train DATASET=SVHN MODE=thermo)"
	@echo "  train-thermo  - Start thermodynamic training (use: make train-thermo DATASET=SVHN)"
	@echo "  train-vanilla - Start vanilla training (use: make train-vanilla DATASET=SVHN)"
	@echo "  train-variational - Start variational training (use: make train-variational DATASET=MNIST)"
	@echo "  sequential    - Run multiple jobs sequentially (use: make sequential CONFIG=jobs.txt)"
	@echo "  distributed   - Run single job on specified device (use: make distributed DATASET=MNIST MODE=thermo DEVICE=0)"
	@echo "  batch         - Run jobs from config in parallel across GPUs (use: make batch CONFIG=jobs.txt NUM_DEVICES=4)"
	@echo "  tune          - Run hyperparameter tuning in vanilla mode (use: make tune DATASET=MNIST)"
	@echo "  baseline      - Train baseline model (use: make baseline MODEL=vae DATASET=CIFAR10)"
	@echo "  baseline-vae  - Train VAE baseline (use: make baseline-vae DATASET=CIFAR10)"
	@echo "  baseline-gan  - Train GAN baseline (use: make baseline-gan DATASET=CIFAR10)"
	@echo "  baseline-ddpm - Train DDPM baseline (use: make baseline-ddpm DATASET=CIFAR10)"
	@echo "  baseline-pang - Train Pang EBM baseline (use: make baseline-pang DATASET=CIFAR10)"
	@echo "  baseline-all  - Train all baselines on dataset (use: make baseline-all DATASET=CIFAR10)"
	@echo "  plot          - Run all plotting scripts"
	@echo "  plot-results  - Run only results plotting scripts"
	@echo "  extract-symbolic - Extract symbolic priors from trained models (use: make extract-symbolic CONFIG=jobs.txt)"
	@echo "  logs          - View latest test log"
	@echo "  clear-logs    - Remove all log files"
	@echo "  julia-setup   - Install Julia dependencies"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Training overview:"
	@echo ""
	@echo "  Command                                               What it does"
	@echo "  ----------------------------------------------------- ------------------------------------"
	@echo "  make train DATASET=X MODE=Y                           Single KAEM job (tmux)"
	@echo "  make baseline MODEL=X DATASET=Y                       Single baseline job (tmux)"
	@echo "  make baseline-all DATASET=Y                           All baselines sequentially (tmux)"
	@echo "  make sequential CONFIG=jobs.txt                       Jobs from file, one at a time"
	@echo "  make distributed DATASET=X MODE=Y DEVICE=N            Single job on specific GPU"
	@echo "  make batch CONFIG=jobs.txt NUM_DEVICES=N              Jobs from file, parallel across GPUs"
	@echo ""
	@echo "Defaults: DATASET=MNIST, MODE=thermo, MODEL=vae, NUM_DEVICES=auto"
	@echo "Datasets: MNIST, FMNIST, CIFAR10, SVHN, CELEBA, PTB, SMS_SPAM, DARCY_FLOW"
	@echo "KAEM Modes: thermo, vanilla, variational"
	@echo "Baseline Modes: baseline-vae, baseline-gan, baseline-ddpm, baseline-pang (use in jobs.txt)"

install:
	@chmod +x scripts/init.sh
	@./scripts/init.sh

uninstall:
	@echo "Removing conda environment..."
	@conda env remove -n $(ENV_NAME) -y 2>/dev/null || echo "Environment not found"
	@echo "Removing Julia project Manifest.toml..."
	@rm -f Manifest.toml
	@echo "Uninstall complete!"

clean:
	@echo "Removing all .log files..."
	@find . -type f -name "*.log" -delete
	@echo "Removing all __pycache__ directories and .pyc files..."
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete
	@echo "Clean complete!"

define conda_run
	@if [ -n "$(CONDA_ACTIVATE)" ]; then \
		. "$(CONDA_ACTIVATE)" && conda activate $(ENV_NAME) && $(1); \
	else \
		echo "Warning: Could not find conda activation script. Trying direct activation..."; \
		conda activate $(ENV_NAME) && $(1); \
	fi
endef

test:
	@mkdir -p logs
	@chmod +x scripts/run_tests.sh
	@tmux kill-session -t kaem_test 2>/dev/null || true
	@tmux new-session -d -s kaem_test -n testing
	@tmux send-keys -t kaem_test:testing "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && ./scripts/run_tests.sh 2>&1 | tee logs/julia_tests_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && ./scripts/run_tests.sh 2>&1 | tee logs/julia_tests_$(shell date +%Y%m%d_%H%M%S).log; fi && tmux kill-session -t kaem_test" Enter
	@echo "Test session started in tmux. Attach with: tmux attach-session -t kaem_test"
	@echo "Log file: logs/julia_tests_$(shell date +%Y%m%d_%H%M%S).log"

bench:
	@mkdir -p logs
	@chmod +x scripts/run_benchmarks.sh
	@tmux kill-session -t kaem_bench 2>/dev/null || true
	@tmux new-session -d -s kaem_bench -n benchmarking
	@tmux send-keys -t kaem_bench:benchmarking "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && ./scripts/run_benchmarks.sh 2>&1 | tee logs/julia_benchmarks_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && ./scripts/run_benchmarks.sh 2>&1 | tee logs/julia_benchmarks_$(shell date +%Y%m%d_%H%M%S).log; fi && tmux kill-session -t kaem_bench" Enter
	@echo "Benchmark session started in tmux. Attach with: tmux attach-session -t kaem_bench"
	@echo "Log file: logs/julia_benchmarks_$(shell date +%Y%m%d_%H%M%S).log"

train:
	@mkdir -p logs
	@echo "Starting training for dataset: $(DATASET), mode: $(MODE)"
	@tmux kill-session -t kaem_train 2>/dev/null || true
	@tmux new-session -d -s kaem_train -n training
	@tmux send-keys -t kaem_train:training "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && DATASET=$(DATASET) MODE=$(MODE) julia --project=. --threads=auto main.jl 2>&1 | tee logs/train_$(MODE)_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && DATASET=$(DATASET) MODE=$(MODE) julia --project=. --threads=auto main.jl 2>&1 | tee logs/train_$(MODE)_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; fi && tmux kill-session -t kaem_train" Enter
	@echo "Training session started in tmux. Attach with: tmux attach-session -t kaem_train"
	@echo "Log file: logs/train_$(MODE)_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log"

train-thermo:
	@$(MAKE) train DATASET=$(DATASET) MODE=thermo

train-vanilla:
	@$(MAKE) train DATASET=$(DATASET) MODE=vanilla

train-variational:
	@$(MAKE) train DATASET=$(DATASET) MODE=variational

tune:
	@mkdir -p logs
	@echo "Starting hyperparameter tuning for dataset: $(DATASET)"
	@tmux kill-session -t kaem_tune 2>/dev/null || true
	@tmux new-session -d -s kaem_tune -n tuning
	@tmux send-keys -t kaem_tune:tuning "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && DATASET=$(DATASET) julia --project=. --threads=1 tuning.jl 2>&1 | tee logs/tune_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && DATASET=$(DATASET) julia --project=. --threads=1 tuning.jl 2>&1 | tee logs/tune_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; fi && tmux kill-session -t kaem_tune" Enter
	@echo "Tuning session started in tmux. Attach with: tmux attach-session -t kaem_tune"
	@echo "Log file: logs/tune_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log"

sequential:
	@mkdir -p logs
	@chmod +x scripts/run_sequential.sh
	@echo "Starting sequential jobs with config: $(CONFIG)"
	@tmux kill-session -t kaem_sequential 2>/dev/null || true
	@tmux new-session -d -s kaem_sequential -n sequential
	@tmux send-keys -t kaem_sequential:sequential "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && ./scripts/run_sequential.sh $(CONFIG) 2>&1 | tee logs/sequential_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && ./scripts/run_sequential.sh $(CONFIG) 2>&1 | tee logs/sequential_$(shell date +%Y%m%d_%H%M%S).log; fi && tmux kill-session -t kaem_sequential" Enter
	@echo "Sequential job session started in tmux. Attach with: tmux attach-session -t kaem_sequential"
	@echo "Log file: logs/sequential_$(shell date +%Y%m%d_%H%M%S).log"

distributed:
	@mkdir -p logs
	@chmod +x scripts/run_distributed.sh
	@echo "Starting training for dataset: $(DATASET), mode: $(MODE), device: $(DEVICE)"
	@tmux kill-session -t kaem_distributed 2>/dev/null || true
	@tmux new-session -d -s kaem_distributed -n distributed
	@tmux send-keys -t kaem_distributed:distributed "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && DATASET=$(DATASET) MODE=$(MODE) DEVICE=$(DEVICE) ./scripts/run_distributed.sh 2>&1 | tee logs/distributed_$(MODE)_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && DATASET=$(DATASET) MODE=$(MODE) DEVICE=$(DEVICE) ./scripts/run_distributed.sh 2>&1 | tee logs/distributed_$(MODE)_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; fi && tmux kill-session -t kaem_distributed" Enter
	@echo "Training session started in tmux. Attach with: tmux attach-session -t kaem_distributed"
	@echo "Log file: logs/distributed_$(MODE)_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log"

batch:
	@mkdir -p logs
	@chmod +x scripts/run_batch.sh
	@echo "Starting batch jobs with config: $(CONFIG), devices: $(NUM_DEVICES)"
	@tmux kill-session -t kaem_batch 2>/dev/null || true
	@tmux new-session -d -s kaem_batch -n batch
	@tmux send-keys -t kaem_batch:batch "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && NUM_WORKERS=$(NUM_DEVICES) ./scripts/run_batch.sh $(CONFIG) 2>&1 | tee logs/batch_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && NUM_WORKERS=$(NUM_DEVICES) ./scripts/run_batch.sh $(CONFIG) 2>&1 | tee logs/batch_$(shell date +%Y%m%d_%H%M%S).log; fi && tmux kill-session -t kaem_batch" Enter
	@echo "Batch session started in tmux. Attach with: tmux attach-session -t kaem_batch"
	@echo "Log file: logs/batch_$(shell date +%Y%m%d_%H%M%S).log"

plot:
	@mkdir -p logs
	@chmod +x scripts/run_plots.sh
	@echo "Running all plotting scripts..."
	@$(call conda_run,./scripts/run_plots.sh 2>&1 | tee logs/plotting_$(shell date +%Y%m%d_%H%M%S).log)
	@echo "Plotting completed! Log file: logs/plotting_$(shell date +%Y%m%d_%H%M%S).log"

plot-results:
	@mkdir -p logs
	@echo "Running results plotting scripts..."
	@$(call conda_run,find plotting/results/ -name "*.py" -exec python {} \; 2>&1 | tee logs/plotting_results_$(shell date +%Y%m%d_%H%M%S).log)
	@echo "Results plotting completed! Log file: logs/plotting_results_$(shell date +%Y%m%d_%H%M%S).log"

extract-symbolic:
	@mkdir -p logs figures/symbolic_priors
	@echo "Extracting symbolic priors from trained models..."
	@echo "Reading jobs from: $(CONFIG)"
	@$(call conda_run,julia --project=. --threads=auto extract_symbolic_priors.jl $(CONFIG) 2>&1 | tee logs/extract_symbolic_$(shell date +%Y%m%d_%H%M%S).log)
	@echo "Finished, figs in figures/symbolic_priors/"

julia-setup:
	@echo "Installing Julia dependencies..."
	@julia --project=. -e "using Pkg; Pkg.instantiate()"
	@echo "Julia dependencies installed!"

logs:
	@if [ -d "logs" ] && [ -n "$$(ls -A logs 2>/dev/null)" ]; then \
		echo "Latest test log:"; \
		ls -t logs/julia_tests_*.log 2>/dev/null | head -1 | xargs cat 2>/dev/null || echo "No test logs found"; \
	else \
		echo "No logs directory or no log files found"; \
	fi

clear-logs:
	@if [ -d "logs" ]; then \
		echo "Removing all log files..."; \
		rm -rf logs/*.log; \
		echo "Logs cleared."; \
	else \
		echo "No logs directory found."; \
	fi

baseline:
	@mkdir -p logs
	@chmod +x scripts/run_distributed.sh
	@echo "Starting baseline training: $(MODEL) on $(DATASET)"
	@tmux kill-session -t kaem_baseline 2>/dev/null || true
	@tmux new-session -d -s kaem_baseline -n baseline
	@tmux send-keys -t kaem_baseline:baseline "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && MODE=baseline-$(MODEL) DATASET=$(DATASET) NUM_WORKERS=$(NUM_DEVICES) ./scripts/run_distributed.sh 2>&1 | tee logs/baseline_$(MODEL)_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && MODE=baseline-$(MODEL) DATASET=$(DATASET) NUM_WORKERS=$(NUM_DEVICES) ./scripts/run_distributed.sh 2>&1 | tee logs/baseline_$(MODEL)_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; fi && tmux kill-session -t kaem_baseline" Enter
	@echo "Baseline training session started in tmux. Attach with: tmux attach-session -t kaem_baseline"
	@echo "Log file: logs/baseline_$(MODEL)_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log"

baseline-vae:
	@$(MAKE) baseline MODEL=vae DATASET=$(DATASET)

baseline-gan:
	@$(MAKE) baseline MODEL=gan DATASET=$(DATASET)

baseline-ddpm:
	@$(MAKE) baseline MODEL=ddpm DATASET=$(DATASET)

baseline-pang:
	@$(MAKE) baseline MODEL=pang DATASET=$(DATASET)

baseline-all:
	@mkdir -p logs
	@chmod +x scripts/run_distributed.sh
	@echo "Starting all baseline training on $(DATASET)"
	@tmux kill-session -t kaem_baseline_all 2>/dev/null || true
	@tmux new-session -d -s kaem_baseline_all -n baseline_all
	@tmux send-keys -t kaem_baseline_all:baseline_all "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && for model in vae gan ddpm pang; do echo \"Training $$model on $(DATASET)...\"; MODE=baseline-$$model DATASET=$(DATASET) NUM_WORKERS=$(NUM_DEVICES) ./scripts/run_distributed.sh 2>&1 | tee -a logs/baseline_all_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; done; else conda activate $(ENV_NAME) && for model in vae gan ddpm pang; do echo \"Training $$model on $(DATASET)...\"; MODE=baseline-$$model DATASET=$(DATASET) NUM_WORKERS=$(NUM_DEVICES) ./scripts/run_distributed.sh 2>&1 | tee -a logs/baseline_all_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; done; fi && tmux kill-session -t kaem_baseline_all" Enter
	@echo "All baselines training session started in tmux. Attach with: tmux attach-session -t kaem_baseline_all"
	@echo "Log file: logs/baseline_all_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log"
