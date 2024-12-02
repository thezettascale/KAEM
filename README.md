# LV-KAM
Official implementation of the Latent Variable Kolmogov-Arnold Model -  a novel generative model defined entirely by finite representations of continous functions.

Go see our [litepaper!](https://exalaboratories.com/litepaper).

## Setup

The shell script will install all requirements. It will install python dependencies into a conda environment called "LV-KAM" for the Python 3.11 scripts. 

```bash
bash setup/setup.sh
```

## Sustainability statement

LV-KAM method does not perform optimally or scale effectively on a GPU. Our hardware offers significantly better performance, currently achieving 27.6x the performance per Watt of GPUs in the [initial public revisions](https://exalaboratories.com/litepaper), with rapid ongoing private improvements and particular suitability to LV-KAM. 

For this study, experiments were conducted on a GPU due to its availability. To mitigate inefficiency as much as possible, we used Julia, which leverages JIT compilation. While initial compilation is slow, Julia excels in performance for repeated evaluations of the same code.