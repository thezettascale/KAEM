# KAEM 

KAEM is a generative model presented [here](https://www.arxiv.org/abs/2506.14167).

## Setup:

Need [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and [Julia](https://github.com/JuliaLang/juliaup). Choose your favourite installer and run: 

```bash
bash <conda-installer-name>-latest-Linux-x86_64.sh
curl -fsSL https://install.julialang.org | sh
```

Then install

```bash
make install
```

[Optional;] Test all Julia scripts:

```bash
make test
```

### Note for windows users:

This repo uses shell scripts solely for convenience, you can run everything without them too. If you want to use the shell scripts, [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) is recommended.

---

## Quick start:

List commands:
```
make help
```

Edit the config files:

```bash
nvim config/nist_config.ini
```

### Training modes

| Command | Description | Defaults |
|---------|-------------|----------|
| `make train DATASET=X MODE=Y` | Single job, single device | DATASET=MNIST, MODE=thermo |
| `make sequential CONFIG=jobs.txt` | Multiple jobs, single device | CONFIG=jobs.txt |
| `make distributed DATASET=X MODE=Y NUM_DEVICES=N` | Single job, multiple devices | NUM_DEVICES=auto |
| `make batch CONFIG=jobs.txt NUM_DEVICES=N` | Multiple jobs, multiple devices | NUM_DEVICES=auto |

Available modes: `vanilla`, `thermo`, `variational`

Shorthand targets:
```bash
make train-vanilla DATASET=MNIST
make train-thermo DATASET=SVHN
make train-variational DATASET=CIFAR10
```

### Sequential jobs

Create a `jobs.txt` file:
```
MNIST thermo
CIFAR10 variational
SVHN vanilla
```

Run sequentially on single device:
```bash
make sequential CONFIG=jobs.txt
```

Run batch with distributed execution (each job uses multiple devices):
```bash
make batch CONFIG=jobs.txt NUM_DEVICES=4
```

### Device configuration

Set device in config files (`config/*.ini`):
```ini
[TRAINING]
device = gpu   # Options: cpu, gpu, tpu
```

For benchmarking run:

```bash
make bench
```

---

## Julia flow:

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

## Samples

### Importance Sampling
> KAEM is a robust probabilistic model. It can even be trained cheaply with importance sampling.

<table>
  <tr>
    <td align="center"><img src="figures/results/individual_plots/mnist_gaussian_rbf.png" width="300"/><br/><b>MNIST</b></td>
    <td align="center"><img src="figures/results/individual_plots/fmnist_gaussian_rbf.png" width="300"/><br/><b>Fashion-MNIST</b></td>
  </tr>
</table>

### Langevin Dynamics
> When importance sampling explodes with variance, unadjusted Langevin algorithm may be used.

<table>
  <tr>
    <td align="center"><img src="figures/results/individual_plots/svhn_real_reference.png" width="300"/><br/><b>SVHN (real)</b></td>
    <td align="center"><img src="figures/results/individual_plots/svhn_vanilla_ula_mixture.png" width="300"/><br/><b>SVHN (generated)</b></td>
  </tr>
</table>

### Thermodynamic Integration
> Annealing as an embarrassingly parallel alternative to diffusion EBMs for improved mixing in latent space.

<table>
  <tr>
    <td align="center"><img src="figures/results/individual_plots/celeba_vanilla_ula_mixture.png" width="300"/><br/><b>CelebA (vanilla)</b></td>
    <td align="center"><img src="figures/results/individual_plots/celeba_thermodynamic_ula_mixture.png" width="300"/><br/><b>CelebA (thermo)</b></td>
  </tr>
</table>


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
