import matplotlib.pyplot as plt
import numpy as np

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

num_param_updates = 16000
num_cycles = 0
initial_p = 1
end_p = 7

x = np.linspace(0, 2 * np.pi * (num_cycles + 0.5), num_param_updates + 1)
p = initial_p + (end_p - initial_p) * 0.5 * (1 - np.cos(x))

plt.figure(figsize=(10, 4))
plt.plot(p)
plt.xlabel("Parameter updates", fontsize=14)
plt.ylabel("p", fontsize=14)
plt.title("Cyclic p schedule", fontsize=14)
# plt.show()
plt.savefig("figures/visual/cyclic_annealing.png", dpi=300)
