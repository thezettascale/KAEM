.PHONY: install uninstall clean test bench train train-thermo train-vanilla train-variational tune sequential distributed batch plot plot-results format lint logs clear-logs julia-setup help

ENV_NAME = KAEM
CONDA_BASE := $(shell conda info --base 2>/dev/null || echo "")
CONDA_ACTIVATE := $(shell if [ -f "$(CONDA_BASE)/etc/profile.d/conda.sh" ]; then echo "$(CONDA_BASE)/etc/profile.d/conda.sh"; elif [ -f "$(CONDA_BASE)/Scripts/activate" ]; then echo "$(CONDA_BASE)/Scripts/activate"; else echo ""; fi)

DATASET ?= MNIST
MODE ?= thermo
NUM_DEVICES ?= auto

XLA_REACTANT_GPU_MEM_FRACTION ?= 0.9
XLA_REACTANT_GPU_PREALLOCATE ?= true
TF_GPU_ALLOCATOR ?= cuda_malloc_async
XLA_FLAGS ?=

export DATASET
export MODE
export NUM_DEVICES
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
	@echo "  sequential    - Schedule multiple jobs sequentially (use: make sequential CONFIG=jobs.txt)"
	@echo "  distributed   - Run distributed training across devices (use: make distributed DATASET=MNIST MODE=thermo NUM_DEVICES=4)"
	@echo "  batch       - Run jobs from config with distributed execution (use: make batch CONFIG=jobs.txt NUM_DEVICES=4)"
	@echo "  tune          - Run hyperparameter tuning in vanilla mode (use: make tune DATASET=MNIST)"
	@echo "  plot          - Run all plotting scripts"
	@echo "  plot-results  - Run only results plotting scripts"
	@echo "  logs          - View latest test log"
	@echo "  clear-logs    - Remove all log files"
	@echo "  julia-setup   - Install Julia dependencies"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Training overview:"
	@echo ""
	@echo "  Command                                               What it does"
	@echo "  ----------------------------------------------------- ------------------------------------"
	@echo "  make train DATASET=X MODE=Y                           Single job, single device"
	@echo "  make sequential CONFIG=jobs.txt                       Multiple jobs, single device"
	@echo "  make distributed DATASET=X MODE=Y NUM_DEVICES=N       Single job, multiple devices"
	@echo "  make batch CONFIG=jobs.txt NUM_DEVICES=N              Multiple jobs, multiple devices"
	@echo ""
	@echo "Defaults: DATASET=MNIST, MODE=thermo, NUM_DEVICES=auto"
	@echo "Datasets: MNIST, FMNIST, CIFAR10, SVHN, CELEBA, CIFAR10PANG, SVHNPANG, CELEBAPANG, PTB, SMS_SPAM, DARCY_FLOW"
	@echo "Modes: thermo, vanilla, variational"

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
	@echo "Starting distributed training for dataset: $(DATASET), mode: $(MODE), devices: $(NUM_DEVICES)"
	@tmux kill-session -t kaem_distributed 2>/dev/null || true
	@tmux new-session -d -s kaem_distributed -n distributed
	@tmux send-keys -t kaem_distributed:distributed "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && DATASET=$(DATASET) MODE=$(MODE) NUM_WORKERS=$(NUM_DEVICES) ./scripts/run_distributed.sh 2>&1 | tee logs/distributed_$(MODE)_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && DATASET=$(DATASET) MODE=$(MODE) NUM_WORKERS=$(NUM_DEVICES) ./scripts/run_distributed.sh 2>&1 | tee logs/distributed_$(MODE)_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; fi && tmux kill-session -t kaem_distributed" Enter
	@echo "Distributed training session started in tmux. Attach with: tmux attach-session -t kaem_distributed"
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
