# LV-KAM
Official implementation of the Latent Variable Kolmogov-Arnold Model -  a novel generative model defined entirely by finite representations of continous functions.

Go see our [litepaper!](https://exalaboratories.com/litepaper).

## Setup

Need [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and [Julia](https://github.com/JuliaLang/juliaup).

The shell script will install all requirements. Python dependencies will be installed into a conda environment called "LV-KAM". Run:

```bash
bash setup/setup.sh
```

[Optional;] Test all Julia scripts:

```bash
bash src/unit_tests/run_tests.sh
```

## Sustainability statement

Like all KANs, LV-KAM neither performs optimally nor scales effectively on a GPU. Our hardware offers significantly better performance, achieving 27.6x the performance per Watt of GPUs in the [initial public revisions](https://exalaboratories.com/litepaper), with rapid ongoing private improvements. LV-KAM is especially well-suited to Exa hardware and will be scaled using such. 

For this initial study however, experiments were conducted on a GPU due to its availability. To mitigate inefficiency as much as possible, Julia was used, which leverages JIT compilation. While initial compilation is slower than PyTorch, Julia excels in performance for repeated evaluations of the same code. Julia is also faster to compile than JAX and easier to read, so it's preferable for prototyping. 

If Julia were a woman, she'd be a queen.

