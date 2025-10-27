# KAEM 

KAEM is a generative model presented [here](https://www.arxiv.org/abs/2506.14167).

## Brief:

The Kolmogorov-Arnold Energy Model (KAEM) is a latent variable model that pairs univariate, energy-based priors with a flexible generator for different data modalities. 

It's been designed to prioritise training stability, inference speed, and interpretability. It can work without an encoder, score-based approximations, and even MCMC (depending on the dataset).

Fast (single forward pass) and unbiased sampling can be feasible with:
- **Inverse transform sampling** from the prior (inference)
- **Importance sampling** for the posterior (training)

<p align="center">
  <img src="figures/results/individual_plots/mnist_ebm_rbf.png" width="25%" />
  <img src="figures/results/individual_plots/fmnist_gaussian_rbf.png" width="25%" />
  <img src="figures/results/individual_plots/darcy_flow_gaussian_fft.png" width="25%" />
</p>

When importance sampling fails, the unadjusted Langevin algorithm (ULA) may be used for posterior sampling instead. Prior sampling can still proceed by inverse transform to preserve fast inference post-training. 

And when ULA and maximum likelihood fail, it can also be trained with a variance-reduction strategy based on Thermodynamic Integration:

<p align="center">
<img src="figures/results/individual_plots/celeba_real_reference.png" width="25%" />
  <img src="figures/results/individual_plots/celeba_vanilla_ula_mixture.png" width="25%" />
  <img src="figures/results/individual_plots/celeba_thermodynamic_ula_mixture.png" width="25%" />
</p>

*This images are after training on a budget with 8,000 parameter updates.

Unlike diffusion and score-based models, annealing is more interpretable, fully parallelizable, and only applied to posterior expectations, (thus preserving inference speed). The main trade-off is expressivity, though this may improve with scaling. And unlike denoising, which scales sequentially, annealing can scale by adding more temperatures in parallel.

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

## Make commands:

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
using .T_KAM_model
using .ModelSetup
using .Utils

model = init_T_KAM(
      dataset, 
      conf, 
      x_shape; 
      file_loc = file_loc, 
      rng = rng
)

# Parse config to setup sampling and training criterions
x, loader_state = iterate(model.train_loader)
x = pu(x)

model, ps, st_kan, st_lux = prep_model(model, x; rng = rng) 
ps_hq = half_quant.(ps) # Mixed precision will return NaN train loss, but grads will be defined

grads = Enzyme.make_zero(ps_hq) # or zero(ps_hq)
loss, grads, st_ebm, st_gen = model.loss_fcn(
      ps_hq,
      grads,
      st_kan,
      st_lux,
      model,
      x;
      train_idx = 1, # Only affects temperature scheduling in thermo model
      rng = Random.default_rng()
)

# States reset with Accessors.jl:
@reset st.ebm = st_ebm
@reset st.gen = st_gen
```

## Citation/license [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The MIT license open-sources the code. The paper is licensed separately with CC license - also open with citation:

```bibtex
@misc{raj2025structuredgenerativemodelingthermodynamic,
      title={Structured Generative Modeling with the Thermodynamic Kolmogorov-Arnold Model}, 
      author={Prithvi Raj},
      year={2025},
      eprint={2506.14167},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.14167}, 
}
```
