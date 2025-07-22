# T-KAM 

T-KAM is a MLE model presented [here.](https://www.arxiv.org/abs/2506.14167)

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

For main experiments run:

```bash
make train-vanilla DATASET=MNIST
make train-thermo DATASET=SVHN
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
      conf, 
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
include("src/utils.jl")
using .T_KAM_model
using .Utils: device, half_quant, full_quant, hq, fq


model = init_T_KAM(
      dataset, 
      conf, 
      x_shape; 
      file_loc = file_loc, 
      rng = rng
)

# Explicit Lux.jl initialisation
params, state = Lux.setup(rng, model) 

# Params must be ComponentArrays.jl. Option to reduce precision
ps = convert(ComponentArray, params) |> hq |> device
st = convert(NamedTuple, state) |> hq |> device

# Parse config to setup sampling and training criterions
model = prep_model(model, params, state, x; rng = rng) 

# Training loss/grads are Reactant.jl compiled
grads = Enzyme.make_zero(ps) # or zero(ps)
loss, grads, st_ebm, st_gen = model.loss_fcn(
      ps,
      grads,
      Lux.trainmode(st),
      model,
      x;
      rng=Random.default_rng()
)

# States reset with Accessors.jl:
@reset st.ebm = st_ebm
@reset st.gen = st_gen
```

## Personal preferences

In this project, implicit types/quantization are never used. Quantization is explicitly declared in function headers using `half_quant` and `full_quant`, defined in [utils.jl](src/utils.jl). Model parameterization is also explicit.

Julia/Lux is adopted instead of PyTorch or JAX due to ‧₊˚✩♡ [substantial personal inclination](https://www.linkedin.com/posts/prithvi-raj-eng_i-moved-from-pytorch-to-jax-to-julia-a-activity-7330842135534919681-9XJF?utm_source=share&utm_medium=member_desktop&rcm=ACoAADUTwcMBFnTsuwtIbYGuiSVLmSAnTVDeOQQ)₊˚✩♡.

The following optimisations are in place:

- Autodifferentiation was switched from [Zygote.jl](https://github.com/FluxML/Zygote.jl) to [Enzyme.jl](https://enzyme.mit.edu/julia/stable/)/[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl/). Enzyme provides highly efficient reverse-mode autodifferentation of statically analyzable LLVM. Reactant compiles to MLIR, (amongst other things).
- Broadcasts, Threads, and CUDA Kernels are now realised with [ParallelStencils.jl](https://github.com/omlins/ParallelStencil.jl). This allows for supremely optimized stencil computations, agnostic to the device in use. 

If there's trouble sourcing cuDNN libraries, the following fix might be applicable:

```bash
export LD_LIBRARY_PATH=$HOME/.julia/artifacts/2eb570b35b597d106228383c5cfa490f4bf538ee/lib:$LD_LIBRARY_PATH
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
