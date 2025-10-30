.PHONY: install uninstall clean test bench train train-thermo train-vanilla plot plot-results format lint logs clear-logs julia-setup help

ENV_NAME = KAEM
CONDA_BASE := $(shell conda info --base 2>/dev/null || echo "")
CONDA_ACTIVATE := $(shell if [ -f "$(CONDA_BASE)/etc/profile.d/conda.sh" ]; then echo "$(CONDA_BASE)/etc/profile.d/conda.sh"; elif [ -f "$(CONDA_BASE)/Scripts/activate" ]; then echo "$(CONDA_BASE)/Scripts/activate"; else echo ""; fi)

# Default values for training
DATASET ?= MNIST
MODE ?= thermo

# Sometimes found trouble sourcing cuDNN libraries, so path exported here
LD_LIB_PATH := $(HOME)/.julia/artifacts/2eb570b35b597d106228383c5cfa490f4bf538ee/lib:$(LD_LIBRARY_PATH)
export LD_LIBRARY_PATH := $(LD_LIB_PATH)

help:
	@echo "Available targets:"
	@echo "  install     - Set up conda environment and install dependencies"
	@echo "  uninstall   - Remove only the dev environment (Conda env and Julia Manifest.toml)"
	@echo "  clean       - Remove conda environment"
	@echo "  test        - Run tests in tmux session with logging"
	@echo "  bench       - Run benchmarks in tmux session with logging"
	@echo "  train       - Start training (use: make train DATASET=SVHN MODE=thermo)"
	@echo "  train-thermo- Start thermodynamic training (use: make train-thermo DATASET=SVHN)"
	@echo "  train-vanilla- Start vanilla training (use: make train-vanilla DATASET=SVHN)"
	@echo "  train-sequential- Schedule multiple training jobs sequentially (use: make train-sequential CONFIG=jobs.txt)"
	@echo "  plot        - Run all plotting scripts"
	@echo "  plot-results- Run only results plotting scripts"
	@echo "  logs        - View latest test log"
	@echo "  clear-logs  - Remove all log files"
	@echo "  format      - Format code"
	@echo "  lint        - Run linting"
	@echo "  julia-setup - Install Julia dependencies"
	@echo "  help        - Show this help"
	@echo ""
	@echo "Available datasets: MNIST, FMNIST, CIFAR10, SVHN, CELEBA, CIFAR10PANG, SVHNPANG, CELEBAPANG, PTB, SMS_SPAM, DARCY_PERM, DARCY_FLOW"
	@echo "Available modes: thermo, vanilla"

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
	@tmux send-keys -t kaem_test:testing "export LD_LIBRARY_PATH=$(LD_LIB_PATH) && if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && ./scripts/run_tests.sh 2>&1 | tee logs/julia_tests_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && ./scripts/run_tests.sh 2>&1 | tee logs/julia_tests_$(shell date +%Y%m%d_%H%M%S).log; fi && tmux kill-session -t kaem_test" Enter
	@echo "Test session started in tmux. Attach with: tmux attach-session -t kaem_test"
	@echo "Log file: logs/julia_tests_$(shell date +%Y%m%d_%H%M%S).log"

bench:
	@mkdir -p logs
	@chmod +x scripts/run_benchmarks.sh
	@tmux kill-session -t kaem_bench 2>/dev/null || true
	@tmux new-session -d -s kaem_bench -n benchmarking
	@tmux send-keys -t kaem_bench:benchmarking "export LD_LIBRARY_PATH=$(LD_LIB_PATH) && if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && ./scripts/run_benchmarks.sh 2>&1 | tee logs/julia_benchmarks_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && ./scripts/run_benchmarks.sh 2>&1 | tee logs/julia_benchmarks_$(shell date +%Y%m%d_%H%M%S).log; fi && tmux kill-session -t kaem_bench" Enter
	@echo "Benchmark session started in tmux. Attach with: tmux attach-session -t kaem_bench"
	@echo "Log file: logs/julia_benchmarks_$(shell date +%Y%m%d_%H%M%S).log"

train:
	@mkdir -p logs
	@echo "Starting training for dataset: $(DATASET), mode: $(MODE)"
	@tmux kill-session -t kaem_train 2>/dev/null || true
	@tmux new-session -d -s kaem_train -n training
ifeq ($(MODE),thermo)
	@tmux send-keys -t kaem_train:training "export LD_LIBRARY_PATH=$(LD_LIB_PATH) && if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && DATASET=$(DATASET) julia --project=. --threads=auto main_thermodynamic.jl 2>&1 | tee logs/train_thermo_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && DATASET=$(DATASET) julia --project=. --threads=auto main_thermodynamic.jl 2>&1 | tee logs/train_thermo_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; fi && tmux kill-session -t kaem_train" Enter
else
	@tmux send-keys -t kaem_train:training "export LD_LIBRARY_PATH=$(LD_LIB_PATH) && if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && DATASET=$(DATASET) julia --project=. --threads=auto main.jl 2>&1 | tee logs/train_vanilla_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && DATASET=$(DATASET) julia --project=. --threads=auto main.jl 2>&1 | tee logs/train_vanilla_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log; fi && tmux kill-session -t kaem_train" Enter
endif
	@echo "Training session started in tmux. Attach with: tmux attach-session -t kaem_train"
ifeq ($(MODE),thermo)
	@echo "Log file: logs/train_thermo_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log"
else
	@echo "Log file: logs/train_vanilla_$(DATASET)_$(shell date +%Y%m%d_%H%M%S).log"
endif

train-thermo:
	@$(MAKE) train DATASET=$(DATASET) MODE=thermo

train-vanilla:
	@$(MAKE) train DATASET=$(DATASET) MODE=vanilla

train-sequential:
	@mkdir -p logs
	@chmod +x scripts/run_sequential_training.sh
	@echo "Starting sequential training with config: $(CONFIG)"
	@tmux kill-session -t kaem_sequential 2>/dev/null || true
	@tmux new-session -d -s kaem_sequential -n sequential
	@tmux send-keys -t kaem_sequential:sequential "export LD_LIBRARY_PATH=$(LD_LIB_PATH) && if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && ./scripts/run_sequential_training.sh $(CONFIG) 2>&1 | tee logs/sequential_training_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && ./scripts/run_sequential_training.sh $(CONFIG) 2>&1 | tee logs/sequential_training_$(shell date +%Y%m%d_%H%M%S).log; fi && tmux kill-session -t kaem_sequential" Enter
	@echo "Sequential training session started in tmux. Attach with: tmux attach-session -t kaem_sequential"
	@echo "Log file: logs/sequential_training_$(shell date +%Y%m%d_%H%M%S).log"

plot:
	@mkdir -p logs
	@chmod +x scripts/run_plots.sh
	@echo "Running all plotting scripts..."
	@export LD_LIBRARY_PATH=$(LD_LIB_PATH) && $(call conda_run,./scripts/run_plots.sh 2>&1 | tee logs/plotting_$(shell date +%Y%m%d_%H%M%S).log)
	@echo "Plotting completed! Log file: logs/plotting_$(shell date +%Y%m%d_%H%M%S).log"

plot-results:
	@mkdir -p logs
	@echo "Running results plotting scripts..."
	@export LD_LIBRARY_PATH=$(LD_LIB_PATH) && $(call conda_run,find plotting/results/ -name "*.py" -exec python {} \; 2>&1 | tee logs/plotting_results_$(shell date +%Y%m%d_%H%M%S).log)
	@echo "Results plotting completed! Log file: logs/plotting_results_$(shell date +%Y%m%d_%H%M%S).log"

format:
	$(call conda_run,black src/ tests/ plotting/)
	$(call conda_run,isort src/ tests/ plotting/)
	@julia --project=. -e 'using JuliaFormatter; format(".")'

lint:
	$(call conda_run,flake8 src/ tests/ plotting/)

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