# LV-KAM
Official implementation of the Latent Variable Kolmogov-Arnold Model - a novel generative model defined entirely by finite representations of continous functions.

Go see our [litepaper!](https://exalaboratories.com/litepaper). We are able to train both MLPs and KANs more efficiently than a GPU, and we can certainly make KANs more efficient than anything else out there.

## What is LV-KAM.

LV-KAM is a generative model that has been **entirely represented in two sums.** It can be visualized, and used without neural networks when preferred, or with neural networks using the ideas in [pykan](https://github.com/KindXiaoming/pykan). 

It uses:

- **The Kolmogorov-Arnold theorem** - any continous function can be represented in a finite manner.
- **Empirical Bayes** - the prior is updated using observations from the data.
- **Inversion/importance sampling** - Markov Chain Monte Carlo is avoided for speed.
- **Thermodynamic Integration** - the practicality of the theorem is improved using another means of marginal likelihood estimation.

## Setup:

Need [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and [Julia](https://github.com/JuliaLang/juliaup). Choose your favourite installer and run: 

```bash
bash <conda-installer-name>-latest-Linux-x86_64.sh
curl -fsSL https://install.julialang.org | sh
```

The [shell script](setup/setup.sh) will install all requirements auto-magically. Python dependencies will be installed into a conda environment called "LV-KAM". Just need to run:

```bash
bash setup/setup.sh
```

[Optional;] Test all Julia scripts:

```bash
bash src/tests/run_tests.sh
```

## To run experiments:

Edit the config files:

```bash
nano nist_config.ini
```

For main experiments run:

```bash
julia nist.jl
```

For benchmarking run:

```bash
bash benchmarking/run_benchmarks.sh
```

## Sustainability statement:

Like all KANs, LV-KAM neither performs optimally nor scales effectively on a GPU. Our hardware offers significantly better performance, achieving **27.6x** the performance per Watt of GPUs in the [initial public revisions](https://exalaboratories.com/litepaper), with rapid ongoing private improvements. LV-KAM is especially well-suited to Exa hardware and will be scaled using such. 

For this initial study however, experiments were conducted on a GPU due to its availability. To mitigate inefficiency as much as possible, [Julia](https://julialang.org/) was used, which leverages [JIT compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation). While initial compilation is slower than PyTorch, Julia excels in performance for repeated evaluations of the same code. Julia is also faster to compile than JAX and easier to read, so it's preferable for prototyping. 

**If Julia were a woman, she'd be a queen.**

