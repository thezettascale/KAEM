import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import magma

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Compute Modern"],
        "axes.unicode_minus": False,
        "text.latex.preamble": (
            r"\usepackage{amsmath} "
            r"\usepackage{amsfonts} "
            r"\usepackage{amssymb} "
            r"\usepackage{bm} "
            r"\newcommand{\probP}{\text{I\kern-0.15em P}}"
        ),
    }
)

z = np.linspace(-5, 5, 500)


def prior(z):
    mean_prior, std_prior = 0, 1
    return np.exp(-0.5 * ((z - mean_prior) / std_prior) ** 2) / (
        std_prior * np.sqrt(2 * np.pi)
    )


def likelihood(z):
    peak1, std1 = 2, 0.8
    peak2, std2 = -1, 0.5
    return 0.6 * np.exp(-0.5 * ((z - peak1) / std1) ** 2) / (
        std1 * np.sqrt(2 * np.pi)
    ) + 0.4 * np.exp(-0.5 * ((z - peak2) / std2) ** 2) / (std2 * np.sqrt(2 * np.pi))


def power_posterior(z, t):
    return prior(z) * (likelihood(z) ** t)


t_values = np.linspace(0, 1, 4)

fig, axes = plt.subplots(1, len(t_values), figsize=(10, 3), sharey=True)
colors = magma(np.linspace(0.9, 0.2, len(t_values)))

for ax, t, color in zip(axes, t_values, colors):
    posterior = power_posterior(z, t)
    posterior /= np.trapz(posterior, z)
    ax.set_xlabel(r"$\bm{\bar{z}}$", fontsize=16)
    ax.plot(z, posterior, color=color, label=rf"$t={t:.2f}$")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=14)

axes[0].set_ylabel(
    r"$\probP(\bm{\bar{z}} \mid \bm{x},\bm{f}, \bm{\alpha}, \bm{\Phi},t)$", fontsize=16
)
plt.ylim(0, 1)
plt.subplots_adjust(hspace=0)
plt.tight_layout()
plt.savefig(
    "figures/visual/power_posterior.png",
    dpi=300,
    bbox_inches="tight",
)
