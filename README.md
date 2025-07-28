# T-KAM 

T-KAM is a generative model presented [here.](https://www.arxiv.org/abs/2506.14167) I'll explain at a later date.

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
vim config/nist_config.ini
```

For individual experiments run:

```bash
make train-vanilla DATASET=MNIST
make train-thermo DATASET=SVHN
```

To automatically run experiments one after the other:
```bash
vim jobs.txt # Schedule jobs
make train-sequential CONFIG=jobs.txt
```

For benchmarking run:

```bash
make bench
```

## Performance tuning and dev preferences

| Stack                                                                    | Reason                                                                                                                                                                                                                                                                                                      | Notes                                                                                                                                                                                                                                                                                                                                                                                                    |
|--------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Julia/Lux.jl](https://github.com/LuxDL/Lux.jl)                                                             | Adopted instead of PyTorch or JAX due to ‧₊˚✩♡ [substantial personal inclination](https://www.linkedin.com/posts/prithvi-raj-eng_i-moved-from-pytorch-to-jax-to-julia-a-activity-7330842135534919681-9XJF?utm_source=share&utm_medium=member_desktop&rcm=ACoAADUTwcMBFnTsuwtIbYGuiSVLmSAnTVDeOQQ) ₊˚✩♡ | Explicitly parameterised, and all functions are strongly typed.                                                                                                                                                                                                                                                                                                                                              |
| [Enzyme.jl](https://enzyme.mit.edu/julia/stable/)                   | Switched from [Zygote.jl](https://github.com/FluxML/Zygote.jl). Enzyme provides much more efficient reverse autodiff of statically analyzable LLVM.                                                                                                                                                    | The next step is [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) which compiles into MLIR.                                                                                                                                                                                                                                                                                                                   |
| [ParallelStencils.jl](https://github.com/omlins/ParallelStencil.jl) | In place of broadcasts, Threads, and CUDA, this enables extraordinarily optimised stencil computations, agnostic to the device in use.                                                                                                                                                                           | Launching CUDA kernels is a host-side action, which Enzyme does not support autodiff through. So any files involved in autodiff have two counterparts; either using stencil loops for CPU parallelization or broadcasts for GPU. CPU parallelization can often outperform GPU for the smaller experiments involving B-splines, inverse transform sampling, or resampling, since search and recursion can cause thread divergence on the GPU. |

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
      img_resize = (16,16), 
      file_loc = loc
)
train!(t)
```

Without trainer:

```julia
using Random, Lux, Enzyme, ComponentArrays, Accessors

include("src/T-KAM/T-KAM.jl")
include("src/T-KAM/model_setup.jl")
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
ps_hq = half_quant.(ps) #Mixed precision

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
