# KAEM

> 🚧 WORK IN PROGRESS 🚧

KAEM is a generative model presented [here](https://www.arxiv.org/abs/2506.14167).

## Setup

Install [Julia](https://github.com/JuliaLang/juliaup) and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -fsSL https://install.julialang.org | sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
uv venv && uv pip install -e ".[dev]"
```

### Note for windows users

This repo uses shell scripts solely for convenience, you can run everything without them too. If you want to use the shell scripts, [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) is recommended.

---

## Quick start

Configure jobs in `jobs.txt`:

```
# KAEM jobs
MNIST thermo
CIFAR10 vanilla
SVHN variational

# Baseline jobs
CIFAR10 baseline-vae
CIFAR10 baseline-gan
CELEBA baseline-ddpm
```

Run:

```bash
./run.sh              # runs jobs.txt
./run.sh other.txt    # runs a different config file
```

Logs are saved to `logs/`.

KAEM modes: `vanilla`, `thermo`, `variational`, `tune`

Baseline modes: `baseline-vae`, `baseline-gan`, `baseline-ddpm`, `baseline-pang`

Datasets: `MNIST`, `FMNIST`, `CIFAR10`, `SVHN`, `CELEBA`, `PTB`, `SMS_SPAM`, `DARCY_FLOW`

### Device configuration

Set device in config files (`config/*.ini`):
```ini
[TRAINING]
device = gpu   # Options: cpu, gpu, tpu
```

### Tests and benchmarks

```bash
./scripts/run_tests.sh
./scripts/run_benchmarks.sh
./scripts/run_plots.sh
```

---

## Julia flow

With trainer (preferable):

```julia
using ConfParser, Random

include("src/pipeline/trainer.jl")
using .trainer

t = init_trainer(
      rng,
      conf, # See config directory for examples
      dataset_name;
      img_resize = (16,16), # Resize for prototyping
      file_loc = loc
)
train!(t)
```

Without trainer:

```julia
using Random, Lux, Enzyme, ComponentArrays, Accessors

include("src/KAEM/KAEM.jl")
include("src/KAEM/model_setup.jl")
include("src/utils.jl")
using .KAEM_model
using .ModelSetup
using .Utils

model = init_KAEM(
      dataset,
      conf,
      x_shape;
      file_loc = file_loc,
      rng = rng
)

# MLIR-compiled loss, (slow to compile, fast to run, see https://mlir.llvm.org/).
x, loader_state = iterate(model.train_loader)
x = pu(x)
model, ps, st_kan, st_lux, st_rng = prep_model(model, x, optimizer; rng = rng)
loss, grads, st_ebm, st_gen = model.loss_fcn(
      ps,
      st_kan,
      st_lux,
      model,
      x,
      st_rng;
      train_idx = 1, # Only affects temperature scheduling in thermo model
)

# States reset with Accessors.jl:
@reset st.ebm = st_ebm
@reset st.gen = st_gen
```
---

## Citation/license [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

```bibtex
@misc{raj2025kolmogorovarnoldenergymodelsfast,
      title={Kolmogorov-Arnold Energy Models: Fast and Interpretable Generative Modeling},
      author={Prithvi Raj},
      year={2025},
      eprint={2506.14167},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.14167},
}
```
