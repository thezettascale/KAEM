# T-KAM 

Official implementation of the Thermodynamic Kolmogov-Arnold Model - a novel generative model defined entirely by finite representations of continous functions.

Go see our [website!](https://exalaboratories.com). We are more efficient and versatile than GPUs/TPUs!

It's early stages and I'm planning to add more - for updates, please follow me on [LinkedIn](https://www.linkedin.com/in/prithvi-raj-eng/) or [Twitter (will add link when password found)]().

## What is T-KAM.

T-KAM is a generative model presented at ...

It uses:

- **The Kolmogorov-Arnold theorem** - any continous function can be represented in a finite manner.
- **Empirical Bayes** - the prior is initialized and updated using observations from the data. It can also be recovered by visualizing its components.
- **Thermodynamic Integration [MAYBE COMING SOON]** - the practicality of the theorem is hopefully going to be improved using another means of marginal likelihood estimation.

## Setup:

Need [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and [Julia](https://github.com/JuliaLang/juliaup). Choose your favourite installer and run: 

```bash
bash <conda-installer-name>-latest-Linux-x86_64.sh
curl -fsSL https://install.julialang.org | sh
```

The [shell script](setup/setup.sh) will install all requirements auto-magically. Python dependencies will be installed into a conda environment called "T-KAM". Just need to run:

```bash
bash setup/setup.sh
```
[Optional;] Install tmux - very useful

```
bash sudo apt install tmux
```

[Optional;] Test all Julia scripts:

```bash
tmux new-session -d -s T_KAM_tests "bash run_tests.sh"
```

## To run experiments:

Edit the config files:

```bash
nano config/nist_config.ini
```

For main experiments run:

```bash
tmux new-session -d -s T_KAM_main "bash run.sh"
```

This starts a tmux session, you can then leave and come back later - touch grass, kiss wife, slap a baby, i dunno.

For benchmarking run:

```bash
tmux new-session -d -s T_KAM_benchmark "bash benches/run_benchmarks.sh"
```

## Personal preferences

In this project, implicit types/quantization are never used. Quantization is explicitly declared in function headers using `half_quant` and `full_quant`, defined in [utils.jl](src/utils.jl). Model parameterization is also explicit.

Julia/Lux is adopted instead of PyTorch or JAX because:

- Julia is faster than Python for repeated code calls, (i.e. training), even against similar Python compilation strategies, (i.e. jax.jit).
- I find Julia easier to prototype with than JAX.
- Julia codes natively to the GPU, (i.e. no separate CUDA/TPU code). Native GPU Python programming was only recently announced. This is important for KANs.

I'm not a code supremacist - these are all subjective/personal preferences that work for me, but may be different for you in different applications. I started off with PyTorch, then JAX, then Julia. I also wish Julia were more popular, hence why I'm trying to spread it. 
