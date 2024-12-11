# LV-KAM
Official implementation of the Latent Variable Kolmogov-Arnold Model -  a novel generative model defined entirely by finite representations of continous functions.

Go see our [litepaper!](https://exalaboratories.com/litepaper).

## Motivation

**AI is blind without guidance** when faced with the poor-quality and scarce datasets of the physical world, and the need for general applicability beyond domains/lab setups.

This is true for all fields beyond generating sentences and images - where our biggest problems lie. Hence why [SciML](https://sites.brown.edu/bergen-lab/research/what-is-sciml/) is emerging, and why [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) is not just a standard, very-large Transformer. 

AI is not a magic wand. Other sucessful statistical models in science and engineering are **derived, then applied** and extended when assumptions break down. AI is backwards in this regard. 

Interpretability allows us to guide models with priors and inductive biases, enabling efficient training and generalization beyond lab setups/domains. Moreover, it helps us uncover new insights from the data, and embed transferable knowledge into future models. 

The point of statistical modeling and data-driven experimentation is not to model the outputs of a lab setup or simply make a decision. We want to learn something new that can be applied to solve a problem. 

**LV-KAM is entirely derived.** It can be visualized, and used without neural networks when preferred, or with neural networks using the ideas in [pykan](https://github.com/KindXiaoming/pykan). 

It uses:

- **The Kolmogorov-Arnold theorem** - any continous function can be represented in a finite manner.
- **Empirical Bayes** - the prior is updated using observations from the data.
- **Rejection/importance sampling** - Markov Chain Monte Carlo is avoided for speed.
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
bash src/unit_tests/run_tests.sh
```

## To run experiments:

Edit the config file:

```bash
nano config.ini
```

Run:

```bash
julia main.jl
```

## Sustainability statement:

Like all KANs, LV-KAM neither performs optimally nor scales effectively on a GPU. Our hardware offers significantly better performance, achieving **27.6x** the performance per Watt of GPUs in the [initial public revisions](https://exalaboratories.com/litepaper), with rapid ongoing private improvements. LV-KAM is especially well-suited to Exa hardware and will be scaled using such. 

For this initial study however, experiments were conducted on a GPU due to its availability. To mitigate inefficiency as much as possible, Julia was used, which leverages JIT compilation. While initial compilation is slower than PyTorch, Julia excels in performance for repeated evaluations of the same code. Julia is also faster to compile than JAX and easier to read, so it's preferable for prototyping. 

**If Julia were a woman, she'd be a queen.**

