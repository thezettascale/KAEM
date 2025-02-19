# T-KAM
Official implementation of the Thermodynamic Kolmogov-Arnold Model - a novel generative model defined entirely by finite representations of continous functions.

Go see our [litepaper!](https://exalaboratories.com/litepaper). We are more efficient and versatile than GPUs/TPUs!

## What is T-KAM.

T-KAM is an alt-gen AI model presented by Exa Laboratories at ...

It uses:

- **The Kolmogorov-Arnold theorem** - any continous function can be represented in a finite manner.
- **Empirical Bayes** - the prior is initialized and updated using observations from the data. It can also be recovered by visualizing its components.
- **Thermodynamic Integration** - the practicality of the theorem is improved using another means of marginal likelihood estimation.

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
bash benchmarking/run_benchmarks.sh
```

Note: I really wanted to do CelebA, but got shooed off by the "Please note that we do not own the copyrights to these images. Their use is RESTRICTED to non-commercial research and educational purposes." 

Unlike some of the other guys in this field, we are not assholes.


