import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import magma

sns.set_style("whitegrid", rc={'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern']})

z = np.linspace(-5, 5, 500)  

def prior(z):
    mean_prior, std_prior = 0, 1
    return np.exp(-0.5 * ((z - mean_prior) / std_prior)**2) / (std_prior * np.sqrt(2 * np.pi))

def likelihood(z):
    peak1, std1 = 2, 0.8
    peak2, std2 = -1, 0.5
    return (
        0.6*np.exp(-0.5 * ((z - peak1) / std1)**2) / (std1 * np.sqrt(2 * np.pi))
        + 0.4*np.exp(-0.5 * ((z - peak2) / std2)**2) / (std2 * np.sqrt(2 * np.pi))
    )

def power_posterior(z, t):
    return prior(z) * (likelihood(z)**t)

t_values = np.linspace(0, 1, 4)  

plt.figure(figsize=(7, 5))
colors = magma(np.linspace(0.9, 0.2, len(t_values)))  

for t, color in zip(t_values, colors):
    posterior = power_posterior(z, t)
    posterior /= np.trapz(posterior, z)  
    plt.plot(z, posterior, label=rf"$t={t:.2f}$", color=color)

plt.title(r"Evolution of power posterior measure on $\mathbf{\bar{z}}$ with $t$", fontsize=14)
plt.xlabel(r"$\mathbf{\bar{z}}$", fontsize=12)
plt.ylabel(r"$\mathcal{P}(\mathbf{\bar{z}} \mid \mathbf{x},\mathbf{f}, \mathbf{\alpha}, \mathbf{\Phi},t)$", fontsize=12)
plt.ylim(0, 1)
plt.legend(title=r"Temperature, $t$", loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
# sns.despine()
plt.savefig("figures/visual/power_posterior.png", dpi=300)
