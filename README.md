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

For individual experiments run:

```bash
make train-vanilla DATASET=MNIST
make train-thermo DATASET=SVHN
```

To automatically run experiments one after the other:
```bash
nvim jobs.txt # Schedule jobs
make train-sequential CONFIG=jobs.txt
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

<p align="center">
  <img src="figures/results/individual_plots/mnist_uniform_rbf.png" alt="MNIST with importance sampling" width="20%" />
  <img src="figures/results/individual_plots/fmnist_gaussian_rbf.png" alt="FMNIST with importance sampling" width="20%" />
</p>

<p align="center">
  <sub>KAEM is a robust probabilistic model. It can even be trained cheaply with importance sampling.</sub>
</p>

<p align="center">
  <img src="figures/results/individual_plots/celeba_vanilla_ula_mixture.png" alt="CelebA with Vanilla training" width="20%" />
  <img src="figures/results/individual_plots/celeba_thermodynamic_ula_mixture.png" alt="CelebaA with Thermodynamic training" width="20%" />
</p>

<p align="center">
  <sub>We present annealing as an embarassingly parallel alternative to diffusion EBMs to improve mixing in the latent space.</sub>
</p>


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
