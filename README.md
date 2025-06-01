# T-KAM 

Official implementation of the Thermodynamic Kolmogov-Arnold Model - a novel generative model defined entirely by finite representations of continous functions.

Go see our [website!](https://exalaboratories.com). We are more efficient and versatile than GPUs/TPUs!

It's early stages and I'm planning to add more - for updates, please follow me on [LinkedIn](https://www.linkedin.com/in/prithvi-raj-eng/) or Twitter (@PritManGuy - link will be added soon).

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

The [shell script](setup/setup.sh) will install all requirements auto-magically. Python dependencies will be installed into a conda environment called "T-KAM", (including [tmux](https://github.com/tmux/tmux/wiki) from conda forge). Just need to run:

```bash
bash setup/setup.sh
```

[Optional;] Test all Julia scripts:

```bash
conda activate T-KAM
tmux new-session -d -s T_KAM_tests "bash run_tests.sh"
```

### Note for windows users:

This repo uses shell scripts solely for convenience and cleanliness, you can run everything without them too. If you want to use the shell scripts, then off the top of my head, do this:

Get [Git Bash](https://gitforwindows.org/), right click the project directory, and click "Git bash here":
 
 ```bash
# Find your path
where conda

# Source it
source /c/path/to/windows/conda.sh

# Activate 
conda activate base

# Run the setup script
bash setup.sh
```

## To run experiments:

Activate conda env:
```
conda activate T-KAM
```

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

Julia/Lux is adopted instead of PyTorch or JAX because.
